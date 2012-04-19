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
Module containing the FMI interface Python wrappers.
"""

import sys
import os
import logging
from operator import itemgetter

import ctypes as C
import numpy as N
from ctypes.util import find_library
import numpy.ctypeslib as Nct
import matplotlib.pyplot as plt

import pyfmi
from pyfmi.common import xmlparser
from pyfmi.common.core import BaseModel, unzip_unit, get_platform_suffix, get_files_in_archive, rename_to_tmp, load_DLL

int = N.int32
N.int = N.int32

"""Flags for evaluation of FMI Jacobians
"""
"""Evaluate Jacobian w.r.t. states."""
FMI_STATES = 1
"""Evaluate Jacobian w.r.t. inputs."""
FMI_INPUTS = 2
"""Evaluate Jacobian of derivatives."""
FMI_DERIVATIVES = 1
"""Evaluate Jacobian of outputs."""
FMI_OUTPUTS = 2

def unzip_fmu(archive, path='.'):
    """
    Unzip an FMU.
    
    Looks for a model description XML file and a binaries directory and returns 
    the result in a dict with the key:value pairs:
            - root : Root of archive (same as path)
            - model_desc : XML description of model (required)
            - image : Image file of model icon (optional)
            - documentation_dir : Directory containing the model documentation (optional)
            - sources_dir : Directory containing source files (optional)
            - binaries_dir : Directory containing the binaries (required)
            - resources_dir : Directory containing resources needed by the model (optional)
    
    
     If the model_desc and/or binaries_dir are not found, an exception will be 
     raised.
    
    Parameters::
        
        archive --
            The archive file name.
            
        path --
            The path to the archive file.
            Default: Current directory.
            
    Raises::
    
        IOError if any file is missing in the FMU.
    """
    tmp_dir = unzip_unit(archive, path)
    fmu_files = get_files_in_archive(tmp_dir)
    
    # check if all obligatory files (but the binary) have been found during unzip
    if fmu_files['model_desc'] == None:
        raise IOError('ModelDescription.xml not found in FMU archive: '+str(archive))
    
    if fmu_files['binaries_dir'] == None:
        raise IOError('binaries folder not found in FMU archive: '+str(archive))
    
    return fmu_files

def unzip_fmux(archive, path='.'):
    """
    Unzip an FMUX.
    
    Looks for a model description XML file and returns the result in a dict with 
    the key words: 'model_desc'. If the file is not found an exception will be 
    raised.
    
    Parameters::
        
        archive --
            The archive file name.
            
        path --
            The path to the archive file.
            Default: Current directory.
            
    Raises::
    
        IOError the model description XML file is missing in the FMU.
    """
    tmpdir = unzip_unit(archive, path)
    fmux_files = get_files_in_archive(tmpdir)
    
    # check if all files have been found during unzip
    if fmux_files['model_desc'] == None:
        raise IOError('ModelDescription.xml not found in FMUX archive: '+str(archive))
    
    return fmux_files

class FMUException(Exception):
    """
    An FMU exception.
    """
    pass


class FMUModel(BaseModel):
    """
    An FMI Model loaded from a DLL.
    """
    
    def __init__(self, fmu, path='.', enable_logging=False):
        """
        Constructor.
        """
        
        # Check that the file referenced by fmu has the correct file-ending
        ext = os.path.splitext(fmu)[1]
        if ext != ".fmu":
            raise FMUException("FMUModel must be instantiated with an FMU (.fmu) file.")
        
        # unzip unit and get files in archive
        self._fmufiles = unzip_fmu(archive=fmu, path=path)
        self._tempxml = self._fmufiles['model_desc']
        
        # Parse XML and set model name (needed when creating temp bin file name)
        self._parse_xml(self._tempxml)
        self._modelid = self.get_identifier()
        self._modelname = self.get_name()
        
        # find model binary in binaries folder and rename to something unique
        suffix = get_platform_suffix()
        if os.path.exists(os.path.join(self._fmufiles['binaries_dir'], self._modelid + suffix)):
            dllname = self._modelid + suffix
        else:
            dllname = self._modelname + suffix
            
        self._tempdll = self._fmufiles['binary'] = rename_to_tmp(dllname, self._fmufiles['binaries_dir'])
        
        #Retrieve and load the binary
        dllname = self._tempdll.split(os.sep)[-1]
        dllname = dllname[:-len(suffix)]
        self._dll = load_DLL(dllname,self._fmufiles['binaries_dir'])
        
        #Load calloc and free
        self._load_c()

        #Set FMIModel Typedefs
        self._set_fmimodel_typedefs()
        
        #Load data from XML file
        self._load_xml()
        
        #Internal values
        self._log = []
        self._enable_logging = enable_logging
        
        #Instantiate
        self.instantiate_model(logging=enable_logging)
        
        #Default values
        self.__t = None
        
        #Internal values
        self._file_open = False
        self._npoints = 0

        #Create a JMIModel if a JModelica generated FMU is loaded
        # This is convenient for debugging purposes
        # Requires uncommenting of the alternative constructor
        # in JMUModel
#        try:
#            self._fmiGetJMI = self._dll.__getattr__('fmiGetJMI')
#            self._fmiInstantiateModel.restype = C.c_voidp()
#            self._fmiGetJMI.argtypes = [self._fmiComponent]
#            self._jmi = self._fmiGetJMI(self._model)
#            self._jmimodel = jmodelica.jmi.JMIModel(self._dll,self._jmi)
#        except:
#            print "Could not create JMIModel"
#            pass

    
    def _load_c(self):
        """
        Loads the C-library and the C-functions 'free' and 'calloc' to
        
            model._free
            model._calloc
        
        Also loads the helper function for the logger into,
        
            model._fmiHelperLogger
        """
        c_lib = C.CDLL(find_library('c'))
        
        self._calloc = c_lib.calloc
        self._calloc.restype = C.c_void_p
        self._calloc.argtypes = [C.c_size_t, C.c_size_t]
        
        self._free = c_lib.free
        self._free.restype = None
        self._free.argtypes = [C.c_void_p]
        
        #Get the path to the helper C function, logger
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),'util') 
        
        #Load the helper function
        if sys.platform == 'win32':
            suffix = '.dll'
        elif sys.platform == 'darwin':
            suffix = '.dylib'
        else:
            suffix = '.so'
        
        cFMILogger = C.CDLL(p+os.sep+'FMILogger'+suffix)        
        
        self._fmiHelperLogger = cFMILogger.pythonCallbacks
        
        
    def _parse_xml(self, fname):
            self._md = xmlparser.ModelDescription(fname)
    
    def _load_xml(self):
        """
        Loads the XML information.
        """
        self._nContinuousStates = self._md.get_number_of_continuous_states()
        self._nEventIndicators = self._md.get_number_of_event_indicators()
        self._GUID = self._md.get_guid()
        self._description = self._md.get_description()
        
        def_experiment = self._md.get_default_experiment()
        if def_experiment != None:
            self._XMLStartTime = self._md.get_default_experiment().get_start_time()
            self._XMLStopTime = self._md.get_default_experiment().get_stop_time()
            self._XMLTolerance = self._md.get_default_experiment().get_tolerance()
            self._tolControlled = True
            
        else:
            self._XMLStartTime = 0.0
            self._XMLTolerance = 1.e-4
            self._tolControlled = False
        
        reals = self._md.get_all_real_variables()
        real_start_values = []
        real_keys = []
        real_names = []
        for real in reals:
            start= real.get_fundamental_type().get_start()
            if start != None:
                real_start_values.append(
                    real.get_fundamental_type().get_start())
                real_keys.append(real.get_value_reference())
                real_names.append(real.get_name())

        self._XMLStartRealValues = N.array(real_start_values,dtype=N.double)
        self._XMLStartRealKeys =   N.array(real_keys,dtype=N.uint32)
        self._XMLStartRealNames =  N.array(real_names)
        
        ints = self._md.get_all_integer_variables()
        int_start_values = []
        int_keys = []
        int_names = []
        for i in ints:
            start = i.get_fundamental_type().get_start()
            if start != None:
                int_start_values.append(i.get_fundamental_type().get_start())
                int_keys.append(i.get_value_reference())
                int_names.append(i.get_name())

        self._XMLStartIntegerValues = N.array(int_start_values,dtype=N.int32)
        self._XMLStartIntegerKeys   = N.array(int_keys,dtype=N.uint32)
        self._XMLStartIntegerNames  = N.array(int_names)
        
        bools = self._md.get_all_boolean_variables()
        bool_start_values = []
        bool_keys = []
        bool_names = []
        for b in bools:
            start = b.get_fundamental_type().get_start()
            if start != None:
                bool_start_values.append(
                    b.get_fundamental_type().get_start())
                bool_keys.append(b.get_value_reference())
                bool_names.append(b.get_name())

        self._XMLStartBooleanValues = N.array(bool_start_values)
        self._XMLStartBooleanKeys   = N.array(bool_keys,dtype=N.uint32)
        self._XMLStartBooleanNames  = N.array(bool_names)
        
        strs = self._md.get_all_string_variables()
        str_start_values = []
        str_keys = []
        str_names = []
        for s in strs:
            start = s.get_fundamental_type().get_start()
            if start != '':
                str_start_values.append(s.get_fundamental_type().get_start())
                str_keys.append(s.get_value_reference())
                str_names.append(s.get_name())

        self._XMLStartStringValues = N.array(str_start_values)
        self._XMLStartStringKeys   = N.array(str_keys,dtype=N.uint32)
        self._XMLStartStringNames  = N.array(str_names)

        for i in xrange(len(self._XMLStartBooleanValues)):
            if self._XMLStartBooleanValues[i] == True:
                if self._md.is_negated_alias(self._XMLStartBooleanNames[i]):
                    self._XMLStartBooleanValues[i] = '0'
                else:
                    self._XMLStartBooleanValues[i] = '1'
            else:
                if self._md.is_negated_alias(self._XMLStartBooleanNames[i]):
                    self._XMLStartBooleanValues[i] = '1'
                else:
                    self._XMLStartBooleanValues[i] = '0'
                
        for i in xrange(len(self._XMLStartRealValues)):
            self._XMLStartRealValues[i] = -1*self._XMLStartRealValues[i] if \
                self._md.is_negated_alias(self._XMLStartRealNames[i]) else \
                self._XMLStartRealValues[i]
        
        for i in xrange(len(self._XMLStartIntegerValues)):
            self._XMLStartIntegerValues[i] = -1*self._XMLStartIntegerValues[i] if \
                self._md.is_negated_alias(self._XMLStartIntegerNames[i]) else \
                self._XMLStartIntegerValues[i]
        
        
        cont_name = []
        cont_valueref = []
        disc_name_r = []
        disc_valueref_r = []
 
        for real in reals:
            if real.get_variability() == xmlparser.CONTINUOUS and \
                real.get_alias() == xmlparser.NO_ALIAS:
                    cont_name.append(real.get_name())
                    cont_valueref.append(real.get_value_reference())
                
            elif real.get_variability() == xmlparser.DISCRETE and \
                real.get_alias() == xmlparser.NO_ALIAS:
                    disc_name_r.append(real.get_name())
                    disc_valueref_r.append(real.get_value_reference())
        
        disc_name_i = []
        disc_valueref_i = []
        for i in ints:
            if i.get_variability() == xmlparser.DISCRETE and \
                i.get_alias() == xmlparser.NO_ALIAS:
                    disc_name_i.append(i.get_name())
                    disc_valueref_i.append(i.get_value_reference())
                    
        disc_name_b = []
        disc_valueref_b =[]
        for b in bools:
            if b.get_variability() == xmlparser.DISCRETE and \
                b.get_alias() == xmlparser.NO_ALIAS:
                    disc_name_b.append(b.get_name())
                    disc_valueref_b.append(b.get_value_reference())

        self._save_cont_valueref = [
            N.array(cont_valueref+disc_valueref_r,dtype=N.uint), 
            disc_valueref_i, 
            disc_valueref_b]
        self._save_cont_name = [cont_name+disc_name_r, disc_name_i, disc_name_b]
        self._save_nbr_points = 0
        self._save_cont_length= [len(self._save_cont_valueref[0]),
                                 len(self._save_cont_valueref[1]),
                                 len(self._save_cont_valueref[2])]
        
    def _set_fmimodel_typedefs(self):
        """
        Connects the FMU to Python by retrieving the C-function by use of ctypes. 
        """
        self._validplatforms = self._dll.__getattr__(
            self._modelid+'_fmiGetModelTypesPlatform')
        self._validplatforms.restype = C.c_char_p
    
        self._version = self._dll.__getattr__(self._modelid+'_fmiGetVersion')
        self._version.restype = C.c_char_p
        
        #Typedefs
        (self._fmiOK,
         self._fmiWarning,
         self._fmiDiscard,
         self._fmiError,
         self._fmiFatal) = map(C.c_int, xrange(5))
        self._fmiStatus = C.c_int
        
        self._fmiComponent = C.c_void_p
        self._fmiValueReference = C.c_uint32
        
        self._fmiReal = C.c_double
        self._fmiInteger = C.c_int32
        self._fmiBoolean = C.c_char
        self._fmiString = C.c_char_p
        self._PfmiString = C.POINTER(self._fmiString)
        
        #Defines
        self._fmiTrue = '\x01'
        self._fmiFalse = '\x00'
        self._fmiUndefinedValueReference = self._fmiValueReference(-1).value
        
        #Struct
        self._fmiCallbackLogger = C.CFUNCTYPE(None, self._fmiComponent, 
            self._fmiString, self._fmiStatus, self._fmiString, self._fmiString)
        self._fmiCallbackAllocateMemory = C.CFUNCTYPE(C.c_void_p, C.c_size_t, 
            C.c_size_t)
        self._fmiCallbackFreeMemory = C.CFUNCTYPE(None, C.c_void_p) 
        
        
        class fmiCallbackFunctions(C.Structure):
            _fields_ = [('logger', self._fmiCallbackLogger),
                        ('allocateMemory', self._fmiCallbackAllocateMemory),
                        ('freeMemory', self._fmiCallbackFreeMemory)]
        
        self._fmiCallbackFunctions = fmiCallbackFunctions
        
        #Sets the types for the helper function
        #--
        self._fmiHelperLogger.restype  = C.POINTER(self._fmiCallbackFunctions)
        self._fmiHelperLogger.argtypes = [self._fmiCallbackFunctions] 
        #--
        
        class fmiEventInfo(C.Structure):
            _fields_ = [('iterationConverged', self._fmiBoolean),
                        ('stateValueReferencesChanged', self._fmiBoolean),
                        ('stateValuesChanged', self._fmiBoolean),
                        ('terminateSimulation', self._fmiBoolean),
                        ('upcomingTimeEvent',self._fmiBoolean),
                        ('nextEventTime', self._fmiReal)]
                        
        class pyEventInfo():
            pass
                        
        self._fmiEventInfo = fmiEventInfo
        self._pyEventInfo = pyEventInfo()
        
        #Methods
        self._fmiInstantiateModel = self._dll.__getattr__(
            self._modelid+'_fmiInstantiateModel')
        self._fmiInstantiateModel.restype = self._fmiComponent
        self._fmiInstantiateModel.argtypes = [self._fmiString, self._fmiString, 
            self._fmiCallbackFunctions, self._fmiBoolean]
        
        self._fmiFreeModelInstance = self._dll.__getattr__(
            self._modelid+'_fmiFreeModelInstance')
        self._fmiFreeModelInstance.restype = C.c_void_p
        self._fmiFreeModelInstance.argtypes = [self._fmiComponent]
        
        self._fmiSetDebugLogging = self._dll.__getattr__(
            self._modelid+'_fmiSetDebugLogging')
        self._fmiSetDebugLogging.restype = C.c_int
        self._fmiSetDebugLogging.argtypes = [
            self._fmiComponent, self._fmiBoolean]
        
        self._fmiSetTime = self._dll.__getattr__(self._modelid+'_fmiSetTime')
        self._fmiSetTime.restype = C.c_int
        self._fmiSetTime.argtypes = [self._fmiComponent, self._fmiReal]
        
        self._fmiCompletedIntegratorStep = self._dll.__getattr__(
            self._modelid+'_fmiCompletedIntegratorStep')
        self._fmiCompletedIntegratorStep.restype = self._fmiStatus
        self._fmiCompletedIntegratorStep.argtypes = [
            self._fmiComponent, C.POINTER(self._fmiBoolean)]
        
        self._fmiInitialize = self._dll.__getattr__(
            self._modelid+'_fmiInitialize')
        self._fmiInitialize.restype = self._fmiStatus
        self._fmiInitialize.argtypes = [self._fmiComponent, self._fmiBoolean, 
            self._fmiReal, C.POINTER(self._fmiEventInfo)]
        
        self._fmiTerminate = self._dll.__getattr__(
            self._modelid+'_fmiTerminate')
        self._fmiTerminate.restype = self._fmiStatus
        self._fmiTerminate.argtypes = [self._fmiComponent]
        
        self._fmiEventUpdate = self._dll.__getattr__(
            self._modelid+'_fmiEventUpdate')
        self._fmiEventUpdate.restype = self._fmiStatus
        self._fmiEventUpdate.argtypes = [self._fmiComponent, self._fmiBoolean, 
            C.POINTER(self._fmiEventInfo)]
        
        self._fmiSetContinuousStates = self._dll.__getattr__(
            self._modelid+'_fmiSetContinuousStates')
        self._fmiSetContinuousStates.restype = self._fmiStatus
        self._fmiSetContinuousStates.argtypes = [self._fmiComponent, 
            Nct.ndpointer(dtype=C.c_double,
                           ndim=1,
                           flags='C') ,C.c_size_t]
        self._fmiGetContinuousStates = self._dll.__getattr__(
            self._modelid+'_fmiGetContinuousStates')
        self._fmiGetContinuousStates.restype = self._fmiStatus
        self._fmiGetContinuousStates.argtypes = [self._fmiComponent, 
            Nct.ndpointer(dtype=C.c_double,
                           ndim=1,
                           flags='C') ,C.c_size_t]
        
        self._fmiGetReal = self._dll.__getattr__(self._modelid+'_fmiGetReal')
        self._fmiGetReal.restype = self._fmiStatus
        self._fmiGetReal.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'),
            C.c_size_t, Nct.ndpointer(dtype=C.c_double,
                           ndim=1,
                           flags='C')]
        self._fmiGetInteger = self._dll.__getattr__(
            self._modelid+'_fmiGetInteger')
        self._fmiGetInteger.restype = self._fmiStatus
        self._fmiGetInteger.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'),
            C.c_size_t, Nct.ndpointer(dtype=C.c_int32,
                                            ndim=1,
                                            flags='C')]
        self._fmiGetBoolean = self._dll.__getattr__(
            self._modelid+'_fmiGetBoolean')
        self._fmiGetBoolean.restype = self._fmiStatus
        self._fmiGetBoolean.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'), 
            C.c_size_t, Nct.ndpointer(dtype=C.c_char,
                           ndim=1,
                           flags='C')]
        self._fmiGetString = self._dll.__getattr__(
            self._modelid+'_fmiGetString')
        self._fmiGetString.restype = self._fmiStatus
        self._fmiGetString.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'), 
            C.c_size_t, self._PfmiString]

        
        self._fmiSetReal = self._dll.__getattr__(self._modelid+'_fmiSetReal')
        self._fmiSetReal.restype = self._fmiStatus
        self._fmiSetReal.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'), 
            C.c_size_t,Nct.ndpointer(dtype=C.c_double,
                           ndim=1,
                           flags='C')]
        self._fmiSetInteger = self._dll.__getattr__(
            self._modelid+'_fmiSetInteger')
        self._fmiSetInteger.restype = self._fmiStatus
        self._fmiSetInteger.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'), 
            C.c_size_t,Nct.ndpointer(dtype=C.c_int32,
                           ndim=1,
                           flags='C')]
        self._fmiSetBoolean = self._dll.__getattr__(
            self._modelid+'_fmiSetBoolean')
        self._fmiSetBoolean.restype = self._fmiStatus
        self._fmiSetBoolean.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'), 
            C.c_size_t,Nct.ndpointer(dtype=C.c_char,
                           ndim=1,
                           flags='C')]
        self._fmiSetString = self._dll.__getattr__(
            self._modelid+'_fmiSetString')
        self._fmiSetString.restype = self._fmiStatus
        self._fmiSetString.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'), 
            C.c_size_t,self._PfmiString]
        
        self._fmiGetDerivatives = self._dll.__getattr__(
            self._modelid+'_fmiGetDerivatives')
        self._fmiGetDerivatives.restype = self._fmiStatus
        self._fmiGetDerivatives.argtypes = [self._fmiComponent, Nct.ndpointer(dtype=C.c_double,
                           ndim=1,
                           flags='C'),
            C.c_size_t]
        
        self._fmiGetEventIndicators = self._dll.__getattr__(
            self._modelid+'_fmiGetEventIndicators')
        self._fmiGetEventIndicators.restype = self._fmiStatus
        self._fmiGetEventIndicators.argtypes = [self._fmiComponent, 
            Nct.ndpointer(dtype=C.c_double,
                           ndim=1,
                           flags='C'), C.c_size_t]
        
        self._fmiGetNominalContinuousStates = self._dll.__getattr__(
            self._modelid+'_fmiGetNominalContinuousStates')
        self._fmiGetNominalContinuousStates.restype = self._fmiStatus
        self._fmiGetNominalContinuousStates.argtypes = [self._fmiComponent, 
            Nct.ndpointer(dtype=C.c_double,
                           ndim=1,
                           flags='C'), C.c_size_t]
        
        self._fmiGetStateValueReferences = self._dll.__getattr__(
            self._modelid+'_fmiGetStateValueReferences')
        self._fmiGetStateValueReferences.restype = self._fmiStatus
        self._fmiGetStateValueReferences.argtypes = [self._fmiComponent, 
            Nct.ndpointer(dtype=C.c_uint32,
                           ndim=1,
                           flags='C'), C.c_size_t]
            
        #self._fmiExtractDebugInfo = self._dll.__getattr__(
        #self._modelid+'_fmiExtractDebugInfo')      
        try:
            self._fmiExtractDebugInfo = self._dll.__getattr__(
                'fmiExtractDebugInfo')
            self._fmiExtractDebugInfo.restype = self._fmiStatus
            self._fmiExtractDebugInfo.argtypes = [self._fmiComponent]
            self._compiled_with_debug_fct = True
        except AttributeError:
            self._compiled_with_debug_fct = False
       
    def _get_time(self):
        return self.__t
    
    def _set_time(self, t):
        t = N.array(t)
        if t.size > 1:
            raise FMUException(
                'Failed to set the time. The size of "t" is greater than one.')
        self.__t = t
        temp = self._fmiReal(t)
        self._fmiSetTime(self._model,temp)
        
    time = property(_get_time,_set_time, doc = 
    """
    Property for accessing the current time of the simulation. Calls the 
    low-level FMI function: fmiSetTime.
    """)
    
    def _get_continuous_states(self):
        values = N.array([0.0]*self._nContinuousStates, dtype=N.double,ndmin=1)
        status = self._fmiGetContinuousStates(
            self._model, values, self._nContinuousStates)
        
        if status != 0:
            raise FMUException('Failed to retrieve the continuous states.')
        
        return values
        
    def _set_continuous_states(self, values):
        values = N.array(values,dtype=N.double,ndmin=1).flatten()
        
        if values.size != self._nContinuousStates:
            raise FMUException(
                'Failed to set the new continuous states. ' \
                'The number of values are not consistent with the number of '\
                'continuous states.')
        
        status = self._fmiSetContinuousStates(
            self._model, values, self._nContinuousStates)
        
        if status >= 3:
            raise FMUException('Failed to set the new continuous states.')
    
    continuous_states = property(_get_continuous_states, _set_continuous_states, 
        doc=
    """
    Property for accessing the current values of the continuous states. Calls 
    the low-level FMI function: fmiSetContinuousStates/fmiGetContinuousStates.
    """)
    
    def _get_nominal_continuous_states(self):
        values = N.array([0.0]*self._nContinuousStates,dtype=N.double,ndmin=1)

        status = self._fmiGetNominalContinuousStates(
            self._model, values, self._nContinuousStates)
        
        if status != 0:
            raise FMUException('Failed to get the nominal values.')
            
        return values
    
    nominal_continuous_states = property(_get_nominal_continuous_states, doc = 
    """
    Property for accessing the nominal values of the continuous states. Calls 
    the low-level FMI function: fmiGetNominalContinuousStates.
    """)
    
    def get_derivatives(self):
        """
        Returns the derivative of the continuous states.
                
        Returns::
        
            dx -- 
                The derivative as an array.
                
        Example::
        
            dx = model.get_derivatives()
                
        Calls the low-level FMI function: fmiGetDerivatives
        """
        values = N.array([0.0]*self._nContinuousStates,dtype=N.double,ndmin=1)

        status = self._fmiGetDerivatives(
            self._model, values, self._nContinuousStates)
        
        if status != 0:
            raise FMUException('Failed to get the derivative values.')
            
        return values
        
    def get_event_indicators(self):
        """
        Returns the event indicators at the current time-point.

        Return::
        
            evInd -- 
                The event indicators as an array.
                
        Example::
        
            evInd = model.get_event_indicators()
                
        Calls the low-level FMI function: fmiGetEventIndicators
        """
        values = N.array([0.0]*self._nEventIndicators,dtype=N.double,ndmin=1)
        status = self._fmiGetEventIndicators(
            self._model, values, self._nEventIndicators)
        
        if status != 0:
            raise FMUException('Failed to get the event indicators.')
            
        return values

    
    def get_tolerances(self):
        """
        Returns the relative and absolute tolerances. If the relative tolerance
        is defined in the XML-file it is used, otherwise a default of 1.e-4 is 
        used. The absolute tolerance is calculated and returned according to
        the FMI specification, atol = 0.01*rtol*(nominal values of the 
        continuous states).
                
        Returns::
        
            rtol -- 
                The relative tolerance.
                
            atol -- 
                The absolute tolerance.
                
        Example::
            
            [rtol, atol] = model.get_tolerances()
        """
        rtol = self._XMLTolerance
        atol = 0.01*rtol*self.nominal_continuous_states
        
        return [rtol, atol]
    
    def event_update(self, intermediateResult=False):
        """
        Updates the event information at the current time-point. If 
        intermediateResult is set to True the update_event will stop at each 
        event iteration which would require to loop until 
        event_info.iterationConverged == fmiTrue.
        
        Parameters::
        
            intermediateResult -- 
                If set to True, the update_event will stop at each event 
                iteration.
                Default: False.
                
        Example::
        
            model.event_update()
        
        Calls the low-level FMI function: fmiEventUpdate
        """
        if intermediateResult:
            status = self._fmiEventUpdate(
                self._model, self._fmiTrue, C.byref(self._eventInfo))
        else:
            status = self._fmiEventUpdate(
                self._model, self._fmiFalse, C.byref(self._eventInfo))
        
        if status != 0:
            raise FMUException('Failed to update the events.')
    
    def save_time_point(self):
        """
        Retrieves the data at the current time-point of the variables defined
        to be continuous and the variables defined to be discrete. The 
        information about the variables are retrieved from the XML-file.
                
        Returns::
        
            sol_real -- 
                The Real-valued variables.
                
            sol_int -- 
                The Integer-valued variables.
                
            sol_bool -- 
                The Boolean-valued variables.
                
        Example::
        
            [r,i,b] = model.save_time_point()
        """
        sol_real=N.array([])
        sol_int=N.array([])
        sol_bool=N.array([])
        if self._save_cont_length[0] > 0:
            sol_real = self.get_real(self._save_cont_valueref[0])
        if self._save_cont_length[1] > 0:
            sol_int  = self.get_integer(self._save_cont_valueref[1])
        if self._save_cont_length[2] > 0:  
            sol_bool = self.get_boolean(self._save_cont_valueref[2])
        
        return sol_real, sol_int, sol_bool
    
    def get_event_info(self):
        """
        Returns the event information from the FMU. 
        
        Returns::
        
            The event information, a struct which contains:
        
            iterationConverged -- 
                Event iteration converged (if True).
                
            stateValueReferencesChanged -- 
                ValueReferences of states x changed (if True).
                
            stateValuesChanged -- 
                Values of states x have changed (if True).
                
            terminateSimulation -- 
                Error, terminate simulation (if True).
                
            upcomingTimeEvent - 
                If True, nextEventTime is the next time event.
                
            nextEventTime -- 
                The next time event.
                
        Example::
            
            event_info    = model.event_info
            nextEventTime = model.event_info.nextEventTime
        """
        
        self._pyEventInfo.iterationConverged          = self._eventInfo.iterationConverged == self._fmiTrue
        self._pyEventInfo.stateValueReferencesChanged = self._eventInfo.stateValueReferencesChanged == self._fmiTrue
        self._pyEventInfo.stateValuesChanged          = self._eventInfo.stateValuesChanged == self._fmiTrue
        self._pyEventInfo.terminateSimulation         = self._eventInfo.terminateSimulation == self._fmiTrue
        self._pyEventInfo.upcomingTimeEvent           = self._eventInfo.upcomingTimeEvent == self._fmiTrue
        self._pyEventInfo.nextEventTime               = self._eventInfo.nextEventTime
        
        return self._pyEventInfo
    
    def get_state_value_references(self):
        """
        Returns the continuous states valuereferences.
                
        Returns::
            
            val -- 
                The references to the continuous states.
                
        Example::
        
            val = model.get_continuous_value_reference()
            
        Calls the low-level FMI function: fmiGetStateValueReferences
        """
        values = N.array([0]*self._nContinuousStates,dtype=N.uint32,ndmin=1)
        status = self._fmiGetStateValueReferences(
            self._model, values, self._nContinuousStates)
        
        if status != 0:
            raise FMUException(
                'Failed to get the continuous state reference values.')
            
        return values
    
    def _get_version(self):
        """
        Returns the FMI version of the Model which it was generated according.
                
        Returns::
        
            version -- 
                The version.
                
        Example::
        
            model.version
        """
        return self._version()
        
    version = property(fget=_get_version)
    
    def _get_model_types_platform(self):
        """
        Returns the set of valid compatible platforms for the Model, extracted
        from the XML.
        
        Returns::
            
            model_types_platform -- 
                The valid platforms.
                
        Example::
        
            model.model_types_platform
        """
        return self._validplatforms()
        
    model_types_platform = property(fget=_get_model_types_platform)
    
    def get_ode_sizes(self):
        """
        Returns the number of continuous states and the number of event 
        indicators.
                
        Returns::
        
            nbr_cont -- 
                The number of continuous states.
                
            nbr_ind -- 
                The number of event indicators.
                
        Example::
        
            [nCont, nEvent] = model.get_ode_sizes()
        """
        return self._nContinuousStates, self._nEventIndicators
    
    def get_real(self, valueref):
        """
        Returns the real-values from the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
        Returns::
        
            values -- 
                The values retrieved from the FMU.
                
        Example::
        
            val = model.get_real([232])
                
        Calls the low-level FMI function: fmiGetReal/fmiSetReal
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = len(valueref)
        values = N.array([0.0]*nref,dtype=N.float, ndmin=1)
        
        status = self._fmiGetReal(self._model, valueref, nref, values)
        
        if status != 0:
            raise FMUException('Failed to get the Real values.')
            
        return values
        
    def set_real(self, valueref, values):
        """
        Sets the real-values in the FMU as defined by the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
            values -- 
                Values to be set.
        
        Example::
        
            model.set_real([234,235],[2.34,10.4])
        
        Calls the low-level FMI function: fmiGetReal/fmiSetReal
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = valueref.size
        values = N.array(values, dtype=N.float, ndmin=1).flatten()

        if valueref.size != values.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        status = self._fmiSetReal(self._model,valueref, nref, values)

        if status != 0:
            raise FMUException('Failed to set the Real values.')
        
    def get_integer(self, valueref):
        """
        Returns the integer-values from the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
        Return::
            
            values -- 
                The values retrieved from the FMU.
                
        Example::
        
            val = model.get_integer([232])
                
        Calls the low-level FMI function: fmiGetInteger/fmiSetInteger
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = len(valueref)
        values = N.array([0]*nref, dtype=int,ndmin=1)
        
        status = self._fmiGetInteger(self._model, valueref, nref, values)
        
        if status != 0:
            raise FMUException('Failed to get the Integer values.')
            
        return values
        
    def set_integer(self, valueref, values):
        """
        Sets the integer-values in the FMU as defined by the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
            values -- 
                Values to be set.
        
        Example::
        
            model.set_integer([234,235],[12,-3])
        
        Calls the low-level FMI function: fmiGetInteger/fmiSetInteger
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = valueref.size
        values = N.array(values, dtype=int,ndmin=1).flatten()
        
        if valueref.size != values.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')
        
        status = self._fmiSetInteger(self._model,valueref, nref, values)
        
        if status != 0:
            raise FMUException('Failed to set the Integer values.')
        
        
    def get_boolean(self, valueref):
        """
        Returns the boolean-values from the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
        Returns::
        
            values -- 
                The values retrieved from the FMU.
                
        Example::
        
            val = model.get_boolean([232])
                
        Calls the low-level FMI function: fmiGetBoolean/fmiSetBoolean
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = len(valueref)
        values = N.array(['0']*nref, dtype=N.char.character,ndmin=1)
        
        status = self._fmiGetBoolean(self._model, valueref, nref, values)
        
        if status != 0:
            raise FMUException('Failed to get the Boolean values.')

        #bol = []
        # char to bol
        #bol = map(lambda x: x == self._fmiTrue, values)
        
        bol = N.array([x==self._fmiTrue for x in values])
        
        return bol
        
    def set_boolean(self, valueref, values):
        """
        Sets the boolean-values in the FMU as defined by the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
            values -- 
                Values to be set.

        Example::
        
            model.set_boolean([234,235],[True,False])
        
        Calls the low-level FMI function: fmiGetBoolean/fmiSetBoolean
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()
        nref = valueref.size
        
        # bool to char
        char_values = map(lambda x: self._fmiTrue if x else self._fmiFalse, values)
        char_values = N.array(char_values, dtype=N.char.character,ndmin=1).flatten()
        
        if valueref.size != char_values.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')
        
        status = self._fmiSetBoolean(self._model,valueref, nref, char_values)
        
        if status != 0:
            raise FMUException('Failed to set the Boolean values.')
        
    def get_string(self, valueref):
        """
        Returns the string-values from the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
        Returns::
        
            values -- 
                The values retrieved from the FMU.
                
        Example::
        
            val = model.get_string([232])
                
        Calls the low-level FMI function: fmiGetString/fmiSetString
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = len(valueref)
        values = N.ndarray([])
        
        temp = (self._fmiString*nref)()
        
        status = self._fmiGetString(self._model, valueref, nref, temp)

        if status != 0:
            raise FMUException('Failed to get the String values.')
        
        return N.array(temp)[:]
    
    def set_string(self, valueref, values):
        """
        Sets the string-values in the FMU as defined by the valuereference(s).
        
        Parameters::
        
            valueref -- 
                A list of valuereferences.
                
            values -- 
                Values to be set.
        
        Example::
        
            model.set_string([234,235],['text','text'])
        
        Calls the low-level FMI function: fmiGetString/fmiSetString
        """
        valueref = N.array(valueref, dtype=N.uint32,ndmin=1).flatten()

        nref = valueref.size
        values = N.array(values).flatten()
        
        temp = (self._fmiString*nref)()
        for i in range(nref):
            temp[i] = values[i]
        
        if valueref.size != values.size:
            raise FMUException(
                'The length of valueref and values are inconsistent.')

        status = self._fmiSetString(self._model, valueref, nref, temp)
        
        if status != 0:
            raise FMUException('Failed to set the String values.')
    
    def set_debug_logging(self,flag):
        """
        Specifies if the debugging should be turned on or off.
        
        Parameters::
        
            flag -- 
                Boolean value.
                
        Calls the low-level FMI function: fmiSetDebuggLogging
        """
        self._enable_logging = flag
        
        if flag:
            status = self._fmiSetDebugLogging(self._model, self._fmiTrue)
        else:
            status = self._fmiSetDebugLogging(self._model, self._fmiFalse)
            
        if status != 0:
            raise FMUException('Failed to set the debugging option.')
        
    
    def get_nominal(self, valueref):
        """
        Returns the nominal value from valueref.
        """
        values = self._xmldoc._xpatheval(
            '//ScalarVariable/Real/@nominal[../../@valueReference=\''+\
            valueref+'\']')
        
        if len(values) == 0:
            return 1.0
        else:
            return float(values[0])
    
    def completed_integrator_step(self):
        """
        This method must be called by the environment after every completed step 
        of the integrator. If the return is True, then the environment must call 
        event_update() otherwise, no action is needed.
        
        Returns::
        
            True -> Call event_update().
            False -> Do nothing.
                
        Calls the low-level FMI function: fmiCompletedIntegratorStep.
        """
        callEventUpdate = self._fmiBoolean(self._fmiFalse)
        status = self._fmiCompletedIntegratorStep(
            self._model, C.byref(callEventUpdate))
        
        if status != 0:
            raise FMUException('Failed to call FMI Completed Step.')
            
        if callEventUpdate.value == self._fmiTrue:
            return True
        else:
            return False
    
    def reset(self):
        """ 
        Calling this function is equivalent to reopening the model.
        """
        #Instantiate
        self.instantiate_model()
        
        #Default values
        self.__t = None
        
        #Internal values
        self._file_open = False
        self._npoints = 0
        self._log = []
    
    def initialize(self, tolControlled=True, relativeTolerance=None):
        """
        Initializes the model and computes initial values for all variables, 
        including setting the start values of variables defined with a the start 
        attribute in the XML-file. 
            
        Parameters::
        
            tolControlled -- 
                If the model are going to be called by numerical solver using
                step-size control. Boolean flag.
            relativeTolerance --
                If the model are controlled by a numerical solver using
                step-size control, the same tolerance should be provided here.
                Else the default tolerance from the XML-file are used.
            
        Calls the low-level FMI function: fmiInitialize.
        """
        #Trying to set the initial time from the xml file, else 0.0
        if self.time == None:
            self.time = self._XMLStartTime
        
        if tolControlled:
            tolcontrolledC = self._fmiBoolean(self._fmiTrue)
            if relativeTolerance == None:
                tol = self._XMLTolerance
            else:
                tol = relativeTolerance
        else:
            tolcontrolledC = self._fmiBoolean(self._fmiFalse)
            tol = self._fmiReal(0.0)
        
        self._eventInfo = self._fmiEventInfo(
            '0','0','0','0','0',self._fmiReal(0.0))
        
        status = self._fmiInitialize(
            self._model, tolcontrolledC, tol, C.byref(self._eventInfo))
        
        if status == 1:
            if self._enable_logging:
                logging.warning(
                    'Initialize returned with a warning.' \
                    ' Check the log for information (FMUModel.get_log).')
            else:
                logging.warning('Initialize returned with a warning.' \
                    ' Enable logging for more information, (FMUModel(..., enable_logging=True)).')
        
        if status > 1:
            raise FMUException('Failed to Initialize the model.')
    
    
    def instantiate_model(self, name='Model', logging=False):
        """
        Instantiate the model.
        
        Parameters::
        
            name -- 
                The name of the instance.
                Default: 'Model'
                        
            logging -- 
                Defines if the logging should be turned on or off.
                Default: False, no logging.
                        
        Calls the low-level FMI function: fmiInstantiateModel.
        """
        instance = self._fmiString(name)
        guid = self._fmiString(self._GUID)
        
        if logging:
            logging_fmi = self._fmiTrue#self._fmiBoolean(self._fmiTrue)
        else:
            logging_fmi = self._fmiFalse#self._fmiBoolean(self._fmiFalse)

        functions = self._fmiCallbackFunctions()#(self._fmiCallbackLogger(self.fmiCallbackLogger),self._fmiCallbackAllocateMemory(self.fmiCallbackAllocateMemory), self._fmiCallbackFreeMemory(self.fmiCallbackFreeMemory))
        
        
        functions.logger = self._fmiCallbackLogger(self.fmiCallbackLogger)
        functions.allocateMemory = self._fmiCallbackAllocateMemory(
            self.fmiCallbackAllocateMemory)
        functions.freeMemory = self._fmiCallbackFreeMemory(
            self.fmiCallbackFreeMemory)
        
        self._functions = functions
        self._modFunctions = self._fmiCallbackFunctions()
        self._modFunctions = self._fmiHelperLogger(self._functions)
        self._modFunctions = self._modFunctions.contents
        
        self._model = self._fmiInstantiateModel(
            instance,guid,self._modFunctions,logging_fmi)
        
        #Just to be safe, some problems with Dymola (2012) FMUs not reacting
        #to logging when set to the instantiate method.    
        self.set_debug_logging(logging)
        
    def fmiCallbackLogger(self,c, instanceName, status, category, message):
        """
        Logg the information from the FMU.
        """
        self._log += [[instanceName, status, category, message]]

    def fmiCallbackAllocateMemory(self, nobj, size):
        """
        Callback function for the FMU which allocates memory needed by the model.
        """
        return self._calloc(nobj,size)

    def fmiCallbackFreeMemory(self, obj):
        """
        Callback function for the FMU which deallocates memory allocated by 
        fmiCallbackAllocateMemory.
        """
        self._free(obj)
    
    def get_log(self):
        """
        Returns the log information as a list. To turn on the logging use the 
        method, set_debug_logging(True). The log is stored as a list of lists. 
        For example log[0] are the first log message to the log and consists of, 
        in the following order, the instance name, the status, the category and 
        the message.
        
        Returns::
        
            log - A list of lists.
        """
        
        # Temporary fix to make the fmi write to logger - remove when 
        # permanent solution is found!
        if self._compiled_with_debug_fct:
            self._fmiExtractDebugInfo(self._model)
        else:
            pass
            #print "FMU not compiled with JModelica.org compliant debug functions"
            #print "Debug info from non-linear solver currently not accessible."
        
        
        return self._log 
    
    def simulate(self,
                 start_time=0.0,
                 final_time=1.0,
                 input=(),
                 algorithm='AssimuloFMIAlg', 
                 options={}):
        """ 
        Compact function for model simulation.
        
        The simulation method depends on which algorithm is used, this can be 
        set with the function argument 'algorithm'. Options for the algorithm 
        are passed as option classes or as pure dicts. See 
        FMUModel.simulate_options for more details.
        
        The default algorithm for this function is AssimuloFMIAlg. 
        
        Parameters::
        
            start_time --
                Start time for the simulation.
                Default: 0.0
                
            final_time --
                Final time for the simulation.
                Default: 1.0
                
            input --
                Input signal for the simulation. The input should be a 2-tuple 
                consisting of first the names of the input variable(s) and then 
                the data matrix.
                Default: Empty tuple.
                
            algorithm --
                The algorithm which will be used for the simulation is specified 
                by passing the algorithm class as string or class object in this 
                argument. 'algorithm' can be any class which implements the 
                abstract class AlgorithmBase (found in algorithm_drivers.py). In 
                this way it is possible to write own algorithms and use them 
                with this function.
                Default: 'AssimuloFMIAlg'
                
            options -- 
                The options that should be used in the algorithm. For details on 
                the options do:
                
                    >> myModel = FMUModel(...)
                    >> opts = myModel.simulate_options()
                    >> opts?

                Valid values are: 
                    - A dict which gives AssimuloFMIAlgOptions with 
                      default values on all options except the ones 
                      listed in the dict. Empty dict will thus give all 
                      options with default values.
                    - An options object.
                Default: Empty dict
        
        Returns::
        
            Result object, subclass of common.algorithm_drivers.ResultBase.
        """
        return self._exec_simulate_algorithm(start_time, 
                                             final_time, 
                                             input,
                                             'pyfmi.fmi_algorithm_drivers', 
                                             algorithm,
                                             options)
                               
    def simulate_options(self, algorithm='AssimuloFMIAlg'):
        """ 
        Get an instance of the simulate options class, prefilled with default 
        values. If called without argument then the options class for the 
        default simulation algorithm will be returned.
        
        Parameters::
        
            algorithm --
                The algorithm for which the options class should be fetched. 
                Possible values are: 'AssimuloFMIAlg'.
                Default: 'AssimuloFMIAlg'
                
        Returns::
        
            Options class for the algorithm specified with default values.
        """
        return self._default_options('pyfmi.fmi_algorithm_drivers', algorithm)
    
    def _set(self, variable_name, value):
        """
        Helper method to set, see docstring on set.
        """
        ref = self.get_valueref(variable_name)
        type = self.get_data_type(variable_name)
        
        if type == 0:  #REAL
            self.set_real([ref], [value])
        elif type == 1: #INTEGER
            self.set_integer([ref], [value])
        elif type == 2: #STRING
            self.set_string([ref], [value])
        elif type == 3: #BOOLEAN
            self.set_boolean([ref], [value])
        else:
            raise FMUException('Type not supported.')
        
    
    def _get(self, variable_name):
        """
        Helper method to get, see docstring on get.
        """
        ref = self.get_valueref(variable_name)
        type = self.get_data_type(variable_name)
        
        if type == 0:  #REAL
            return self.get_real([ref])
        elif type == 1: #INTEGER
            return self.get_integer([ref])
        elif type == 2: #STRING
            return self.get_string([ref])
        elif type == 3: #BOOLEAN
            return self.get_boolean([ref])
        else:
            raise FMUException('Type not supported.')
    
    #XML PART
    def get_variable_descriptions(self, include_alias=True):
        """
        Extract the descriptions of the variables in a model.

        Returns::
        
            Dict with ValueReference as key and description as value.
        """
        return self._md.get_variable_descriptions(include_alias)
        
    def get_data_type(self, variablename):
        """ 
        Get data type of variable. 
        """
        return self._md.get_data_type(variablename)
        
    def get_valueref(self, variablename=None, type=None):
        """
        Extract the ValueReference given a variable name.
        
        Parameters::
        
            variablename -- 
                The name of the variable.
            
        Returns::
        
            The ValueReference for the variable passed as argument.
        """
        if variablename != None:
            return self._md.get_value_reference(variablename)
        else:
            valrefs = []
            allvariables = self._md.get_model_variables()
            for variable in allvariables:
                if variable.get_variability() == type:
                    valrefs.append(variable.get_value_reference())
                    
        return N.array(valrefs,dtype=N.int)
    
    def get_variable_names(self, type=None, include_alias=True):
        """
        Extract the names of the variables in a model.

        Returns::
        
            Dict with variable name as key and value reference as value.
        """
        
        if type != None:
            variables = self._md.get_model_variables()
            names = []
            valuerefs = []
            if include_alias:
                for var in variables:
                    if var.get_variability()==type:
                        names.append(var.get_name())
                        valuerefs.append(var.get_value_reference())
                return zip(tuple(vrefs), tuple(names))
            else:
                for var in variables: 
                    if var.get_variability()==type and \
                        var.get_alias() == xmlparser.NO_ALIAS:
                            names.append(var.get_name())
                            valuerefs.append(var.get_value_reference())
                return zip(tuple(vrefs), tuple(names))
        else:
            return self._md.get_variable_names(include_alias)
    
    def get_alias_for_variable(self, aliased_variable, ignore_cache=False):
        """ 
        Return list of all alias variables belonging to the aliased variable 
        along with a list of booleans indicating whether the alias variable 
        should be negated or not.

        Returns::
        
            A list consisting of the alias variable names and another list 
            consisting of booleans indicating if the corresponding alias is 
            negated.
            
        Raises:: 
        
            XMLException if alias_variable is not in model.
        """
        return self._md.get_aliases_for_variable(aliased_variable)
    
    def get_variable_aliases(self):
        """
        Extract the alias data for each variable in the model.
        
        Returns::
        
            A list of tuples containing value references and alias data 
            respectively.
        """
        return self._md.get_variable_aliases()
        
    def get_variability(self, variablename):
        """ 
        Get variability of variable. 
            
        Parameters::
        
            variablename --
            
                The name of the variable.
                    
        Returns::
        
            The variability of the variable, CONTINUOUS(0), CONSTANT(1), 
            PARAMETER(2) or DISCRETE(3)
        """
        return self._md.get_variability(variablename)
    
    def get_name(self):
        """ 
        Return the model name as used in the modeling environment.
        """
        return self._md.get_model_name()
    
    def get_identifier(self):
        """ 
        Return the model identifier, name of binary model file and prefix in 
        the C-function names of the model. 
        """
        return self._md.get_model_identifier()
    
    def __del__(self):
        """
        Destructor.
        """
        import os
        import sys
        
        #Deallocate the models allocation
        self._fmiTerminate(self._model)
        
        #--ERROR
        if sys.platform == 'win32':
            try:
                self._fmiFreeModelInstance(self._model)
            except WindowsError:
                print 'Failed to free model instance.'
        else:
            self._fmiFreeModelInstance(self._model)
           
        #Remove the temporary xml
        os.remove(self._tempxml)
        #Remove the temporary binary
        try:
            os.remove(self._tempdll)
        except:
            print 'Failed to remove temporary dll ('+ self._tempdll+').'

class FMUModel2(FMUModel):
    """
    An FMI 2.0 Model loaded from a DLL.
    
    This class is still very experimental and is used for prototyping
    """

    def _parse_xml(self, fname):
            self._md = xmlparser.ModelDescription2(fname)

    def _set_fmimodel_typedefs(self):
        """
        Connects the FMU to Python by retrieving the C-function by use of ctypes. 
        """
        
        FMUModel._set_fmimodel_typedefs(self)
                
        try:
            self._fmiGetJacobian = self._dll.__getattr__(self._modelid+'_fmiGetJacobian')
            self._fmiGetJacobian.restype = self._fmiStatus
            self._fmiGetJacobian.argtypes = [self._fmiComponent, self._fmiInteger,
                                             self._fmiInteger, Nct.ndpointer(),
                                             C.c_size_t]

            self._fmiGetDirectionalDerivative = self._dll.__getattr__(self._modelid+'_fmiGetDirectionalDerivative')
            self._fmiGetDirectionalDerivative.restype = self._fmiStatus
            self._fmiGetDirectionalDerivative.argtypes = [self._fmiComponent,
                                                          Nct.ndpointer(), C.c_size_t,
                                                          Nct.ndpointer(), C.c_size_t,
                                                          Nct.ndpointer(), Nct.ndpointer()]

        except:
            pass
 
        # A struct that hods a raw double vector and the size
        # of the matrix. Row major format is used by default.
        class fmiJacobianStruct(C.Structure):
            _fields_ = [('data', C.POINTER(self._fmiReal)),
                        ('nRow', self._fmiInteger),
                        ('nCol', self._fmiInteger)]

        self._fmiJacobianStruct = fmiJacobianStruct
 
        self._fmiSetMatrixElementType = C.CFUNCTYPE(C.c_int32, C.POINTER(self._fmiJacobianStruct), self._fmiInteger,
                                                        self._fmiInteger, self._fmiReal)        

        self._fmiGetPartialDerivatives = self._dll.__getattr__(
        self._modelid+'_fmiGetPartialDerivatives')
        self._fmiGetPartialDerivatives.restype = self._fmiStatus
        self._fmiGetPartialDerivatives.argtypes = [self._fmiComponent,
                                                   self._fmiSetMatrixElementType,
                                                   C.c_void_p,
                                                   C.c_void_p,
                                                   C.c_void_p,
                                                   C.c_void_p]
        
    def fmiSetMatrixElement(self, matrix, row, col, value):
        """
        Element setter function that is passed into the FMU.
        
        Parameters::
        
            matrix --
                Pointer to a the user defined data structure containing the matrix.
            
            row --
                Row index.
                
            col --
                Cloumn index.
                
            value --
                Value of the matrix entry.
        """

        #Access struct contents via contents
        nRow = matrix.contents.nRow
        nCol = matrix.contents.nCol
        #Access double* data
        data = matrix.contents.data

        
        #print "Nrow, Ncol, row, col, ind, value", nRow, nCol, row, col, (row-1)*nRow + (col-1), value
                
        #for i in range(nRow):
        #    for j in range(nCol):
        #        print "<<< " + repr(data[(i)*nCol+(j)])

        # Set the data
        data[(row-1)*nCol + (col-1)] = value

        return 0

    def get_partial_derivatives(self):
        """
        Evaluate the Jacobian matrices A, B, C, D of the ODE.
        
        This function evaluates one or several of the A, B, C, D 
        Jacobian matrices in the linearized ODE:
        
          dx = A*x + B*u
          y = C*x + D*u
        
        where dx are the derivatives, u are the continuous inputs, 
        y are the continuous top level outputs and x are the states.
        
        Returns::
           
            A, B, C, D --
                The Jacobian matrices.
        
        """
        
        # Retrieve sizes
        nx = self._md.get_number_of_continuous_states()
        ncu = self._md.get_number_of_continuous_inputs()
        ncy = self._md.get_number_of_continuous_outputs()
        
        # Create Numpy arrays and then extract the data vectors
        # and insert these into the structs
        Ajac = N.zeros((nx,nx))
        Am = self._fmiJacobianStruct()
        Am.data = C.cast(Ajac.ctypes.data, C.POINTER(C.c_double))
        Am.nRow = nx
        Am.nCol = nx

        Bjac = N.zeros((nx,ncu))
        Bm = self._fmiJacobianStruct()
        Bm.data = C.cast(Bjac.ctypes.data, C.POINTER(C.c_double))
        Bm.nRow = nx
        Bm.nCol = ncu

        Cjac = N.zeros((ncy,nx))
        Cm = self._fmiJacobianStruct()
        Cm.data = C.cast(Cjac.ctypes.data, C.POINTER(C.c_double))
        Cm.nRow = ncy
        Cm.nCol = nx

        Djac = N.zeros((ncy,ncu))
        Dm = self._fmiJacobianStruct()
        Dm.data = C.cast(Djac.ctypes.data, C.POINTER(C.c_double))
        Dm.nRow = ncy
        Dm.nCol = ncu
        
        # Evaluate the Jacobians
        status = self._fmiGetPartialDerivatives(self._model, self._fmiSetMatrixElementType(self.fmiSetMatrixElement), 
                                       C.byref(Am),C.byref(Bm),C.byref(Cm),C.byref(Dm))
      
        if status != 0:
            raise FMUException('Failed to evaluate the Jacobian.')

        return (Ajac,Bjac,Cjac,Djac)

    def get_partial_derivatives_incidence(self):
        """
        Get the sparsity patterns of the Jacobian matrices. The patterns
        are returned in matrices where each row contains the row and column
        indices (0-indexing) of the non-zero elements. One matrix is returned
        for each of the Jacobians A, B, C, D.
        
        Returns::
        
            A tuple with four matrices containing the sparsity patterns
            for A, B, C, and D respectively.
        """
        A_irow = []
        A_icol = []
        B_irow = []
        B_icol = []
        C_irow = []
        C_icol = []
        D_irow = []
        D_icol = []
        
        irow = 0
        for v in self._md._model_structure_derivatives:
            for icol in v.get_state_dependency():
                A_irow.append(irow)
                A_icol.append(icol)
            irow = irow + 1
           
        A_st = N.zeros((len(A_icol),2),dtype=int)
        A_st[:,0] = A_irow
        A_st[:,1] = A_icol
        
        irow = 0
        for v in self._md._model_structure_derivatives:
            for icol in v.get_input_dependency():
                B_irow.append(irow)
                B_icol.append(icol)
            irow = irow + 1
           
        B_st = N.zeros((len(B_icol),2),dtype=int)
        B_st[:,0] = B_irow
        B_st[:,1] = B_icol

        irow = 0
        for v in self._md._model_structure_outputs:
            for icol in v.get_state_dependency():
                C_irow.append(irow)
                C_icol.append(icol)
            irow = irow + 1
           
        C_st = N.zeros((len(C_icol),2),dtype=int)
        C_st[:,0] = C_irow
        C_st[:,1] = C_icol

        irow = 0
        for v in self._md._model_structure_outputs:
            for icol in v.get_input_dependency():
                D_irow.append(irow)
                D_icol.append(icol)
            irow = irow + 1
           
        D_st = N.zeros((len(D_icol),2),dtype=int)
        D_st[:,0] = D_irow
        D_st[:,1] = D_icol

        return (A_st,B_st,C_st,D_st)

    def get_jacobian(self, independents, dependents, jac):
        """
        Evaluate Jacobian(s) of the ODE 
        (ONLY FOR JMODELICA.ORG FMUS.)

        This function evaluates one or several of the A, B, C, D Jacobian matrices in the
        linearized ODE:
        
          dx = A*x + B*u
          y = C*x + D*u
        
        where dx are the derivatives, u are the inputs, y are the top level outputs and
        x are the states. The arguments 'independents' and 'dependents' are used to
        specify which Jacobian(s) to compute: independents=FMI_STATES and
        dependents=FMI_DERIVATIVES gives the A matrix, and
        independents=FMI_STATES|FMI_INPUTS and dependents=FMI_DERIVATIVES|FMI_OUTPUTS
        gives the A, B, C and D matrices in block matrix form:
        
          A  |  B
          -------
          C  |  D

        Parameters::

            independents --
                Should be FMI_STATES and/or FMI_INPUTS.

            dependents --
                Should be be FMI_DERIVATIVES and/or FMI_OUTPUTS.

            jac --
                A vector representing a matrix on column major format.

        Example::
        
            jac = model.get_jacobian(jmodelica.fmi.FMI_STATES,
                                     jmodelica.fmi.FMI_DERIVATIVES, jac)
                
        Calls the low-level FMI function: fmiGetJacobian
        """
        status = self._fmiGetJacobian(
            self._model, independents, dependents, jac, len(jac))
        
        if status != 0:
            raise FMUException('Failed to evaluate the Jacobian.')

    def get_directional_derivative(self, z_vref, v_vref, dz, dv):
        """
        Evaluate directional derivative of the ODE. 
        (ONLY FOR JMODELICA.ORG FMUS)

        Paramters::

            z_vref --
                Value references of the directional derivative result vector dz.
                These are defined by a subset of the derivative and output variable
                value references.

            v_ref --
                Value reference of the input seed vector dv. These are defined by a
                subset of the state and input variable value references.

            dz --
                Output argument containing the directional derivative vector.

            dv -- Input argument containing the input seed vector.
                
        Calls the low-level FMI function: fmiGetDirectionalDerivative
        """
        status = self._fmiGetDirectionalDerivative(
            self._model, z_vref, len(z_vref), v_vref, len(v_vref), dz, dv)
        
        if status != 0:
            raise FMUException('Failed to evaluate the directional derivative.')

    def check_jacobians(self, delta_abs=1e-2, delta_rel=1e-6, tol=1e-3, 
                        plot_sparsity_check=False):
        """
        Check if the Jacobians are correct by means of finite differences.
        
        The increment used in the finite difference is computed according to
        
        delta_i = (z_i + delta_abs)*delta_rel
        
        where z_i is a state or input for which the finite difference is
        computed. 
        
        An error in a Jacobian entry is reported if the relative error is
        larger than tol.
        
        Parameters::
        
            delta_abs --
                Absolute delta used in computation of finite difference increment.
                
            delta_rel --
                Relative delta used in computation of finite difference increment.
                
            tol --
                Tolerance for detecting Jacobian errors. 
                
            plot_sparsity_check --
                Generate plots for the sparsity patterns of the A, B, C, and D
                matrices. A black 'x' indicates that this entry is present in 
                the variable dependency information in the XML file. A red '+'
                indicates that the corresponding entry is non-zero i the evaluated
                Jacobian.
                
        """
        A,B,C,D = self.get_partial_derivatives()
        A_st,B_st,C_st,D_st = self.get_partial_derivatives_incidence()
        
        nx = self._md.get_number_of_continuous_states()
        ncu = self._md.get_number_of_continuous_inputs()
        ncy = self._md.get_number_of_continuous_outputs()

        Afd = N.zeros((nx,nx))
        Bfd = N.zeros((nx,ncu))
        Cfd = N.zeros((ncy,nx))
        Dfd = N.zeros((ncy,ncu))

        x = self.continuous_states
    
        yc_vrefs = self._md.get_continous_outputs_value_references()
        uc_vrefs = self._md.get_continous_inputs_value_references()    
    
        for i in range(nx):
            if x[i] < 0:
                delta = (x[i] - delta_abs)*delta_rel
            else:
                delta = (x[i] + delta_abs)*delta_rel

            #print delta
            x[i] = x[i] + delta
            self.continuous_states = x
            fpA = self.get_derivatives()
            fpC = self.get_real(yc_vrefs)

            x[i] = x[i] - 2*delta
            self.continuous_states = x
            fnA = self.get_derivatives()
            fnC = self.get_real(yc_vrefs)

            Afd[:,i] = (fpA-fnA)/2./delta
            Cfd[:,i] = (fpC-fnC)/2./delta

            x[i] = x[i] + delta
            self.continuous_states = x

        u = self.get_real(uc_vrefs)
    
        for i in range(ncu):
            if u[i] < 0:
                delta = (u[i] - delta_abs)*delta_rel
            else:
                delta = (u[i] + delta_abs)*delta_rel

            u[i] = u[i] + delta
            self.set_real(uc_vrefs,u)
            fpB = self.get_derivatives()
            fpD = self.get_real(yc_vrefs)

            u[i] = u[i] - 2*delta
            self.set_real(uc_vrefs,u)
            fnB = self.get_derivatives()
            fnD = self.get_real(yc_vrefs)

            Bfd[:,i] = (fpB-fnB)/2./delta
            Dfd[:,i] = (fpD-fnD)/2./delta

            u[i] = u[i] + delta
            self.set_real(uc_vrefs,u)
            

        n_err = 0
        print "Errors in Jaobians:"
        for i in range(nx):
            for j in range(nx):
                if N.abs((A[i,j]-Afd[i,j])/(N.abs(Afd[i,j]) + 1)) > tol:                    
                    print "A[" + repr(i).rjust(3) + "," + repr(j).rjust(3) + \
                    "] - jac: " +"{0: e}".format(A[i,j]) + \
                    " - fd: " + "{0: e}".format(Afd[i,j]) + \
                    " - err: " + "{0: e}".format(A[i,j]-Afd[i,j])
                    n_err = n_err + 1
                    

        for i in range(nx):
            for j in range(ncu):
                if N.abs((B[i,j]-Bfd[i,j])/(N.abs(Bfd[i,j]) + 1)) > tol:
                    print "B[" + repr(i).rjust(3) + "," + repr(j).rjust(3) + \
                    "] - jac: " +"{0: e}".format(B[i,j]) + \
                    " - fd: " + "{0: e}".format(Bfd[i,j]) + \
                    " - err: " + "{0: e}".format(B[i,j]-Bfd[i,j])  
                    n_err = n_err + 1

        for i in range(ncy):
            for j in range(nx):
                if N.abs((C[i,j]-Cfd[i,j])/(N.abs(Cfd[i,j]) + 1)) > tol:
                    print "C[" + repr(i).rjust(3) + "," + repr(j).rjust(3) + \
                    "] - jac: " +"{0: e}".format(C[i,j]) + \
                    " - fd: " + "{0: e}".format(Cfd[i,j]) + \
                    " - err: " + "{0: e}".format(C[i,j]-Cfd[i,j])  
                    n_err = n_err + 1

        for i in range(ncy):
            for j in range(ncu):
                if N.abs((D[i,j]-Dfd[i,j])/(N.abs(Dfd[i,j]) + 1)) > tol:
                    print "D[" + repr(i).rjust(3) + "," + repr(j).rjust(3) + \
                    "] - jac: " +"{0: e}".format(D[i,j]) + \
                    " - fd: " + "{0: e}".format(Dfd[i,j]) + \
                    " - err: " + "{0: e}".format(D[i,j]-Dfd[i,j])  
                    n_err = n_err + 1

        print "Found " + repr(n_err) + " errors"

        if plot_sparsity_check:
            
            h = plt.figure(1)
            plt.clf()
            h.gca().set_ylim((nx+1,0))
            plt.plot(A_st[:,1]+1,A_st[:,0]+1,'kx')
            plt.hold(True)

            for i in range(nx):
                for j in range(nx):
                    if (N.abs(A[i,j])>1e-8):
                        plt.plot(j+1,i+1,'r+')
            plt.axis([0, nx+1, nx+1, 0])
            
            plt.show()

            h = plt.figure(2)
            plt.clf()
            h.gca().set_ylim((nx+1,0))
            plt.plot(B_st[:,1]+1,B_st[:,0]+1,'kx')
            plt.hold(True)

            for i in range(nx):
                for j in range(ncu):
                    if (N.abs(B[i,j])>1e-8):
                        plt.plot(j+1,i+1,'r+')
            plt.axis([0, ncu+1, nx+1, 0])
            
            plt.show()

            h = plt.figure(3)
            plt.clf()
            h.gca().set_ylim((ncy+1,0))
            plt.plot(C_st[:,1]+1,C_st[:,0]+1,'kx')
            plt.hold(True)

            for i in range(ncy):
                for j in range(nx):
                    if (N.abs(C[i,j])>1e-8):
                        plt.plot(j+1,i+1,'r+')
            plt.axis([0, nx+1, ncy+1, 0])
            
            plt.show()

            h = plt.figure(4)
            plt.clf()
            h.gca().set_ylim((ncy+1,0))            
            plt.plot(D_st[:,1]+1,D_st[:,0]+1,'kx')
            plt.hold(True)

            for i in range(ncy):
                for j in range(ncu):
                    if (N.abs(D[i,j])>1e-8):
                        plt.plot(j+1,i+1,'r+')
            plt.axis([0, ncu+1, ncy+1, 0])
            
            plt.show()
            
        return Afd,Bfd,Cfd,Dfd,n_err
    
