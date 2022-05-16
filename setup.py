#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2022 Modelon AB
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

#from distutils.core import setup, Extension
#from distutils.ccompiler import new_compiler


import distutils
import os as O
import sys as S
import shutil
import numpy as N
import ctypes.util
import sys

#If prefix is set, we want to allow installation in a directory that is not on PYTHONPATH
#and this is only possible with distutils, not setuptools
if str(sys.argv[1:]).find("--prefix") == -1:
    from setuptools import setup  
else:
    from distutils.core import setup

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    raise Exception("Please upgrade to a newer Cython version, >= 0.15.")


NAME = "PyFMI"
AUTHOR = "Modelon AB"
AUTHOR_EMAIL = ""
VERSION = "2.9.7"
LICENSE = "LGPL"
URL = "https://jmodelica.org/pyfmi"
DOWNLOAD_URL = "https://jmodelica.org/pyfmi/installation.html"
DESCRIPTION = "A package for working with dynamic models compliant with the Functional Mock-Up Interface standard."
PLATFORMS = ["Linux", "Windows", "MacOS X"]
CLASSIFIERS = [ 'Programming Language :: Python',
                'Operating System :: MacOS :: MacOS X',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: Unix']

LONG_DESCRIPTION = """
PyFMI is a package for loading and interacting with Functional Mock-Up 
Units (FMUs), which are compiled dynamic models compliant with the 
Functional Mock-Up Interface (FMI), see 
https://www.fmi-standard.org/ for more information. PyFMI
is based on FMI Library, see https://github.com/modelon-community/fmi-library .

FMI is a standard that enables tool independent exchange of dynamic 
models on binary format. Several industrial simulation platforms 
supports export of FMUs, including, Impact, Dymola, OpenModelica 
and SimulationX, see https://www.fmi-standard.org/tools 
for a complete list. PyFMI offers a Python interface for interacting 
with FMUs and enables for example loading of FMU models, setting of 
model parameters and evaluation of model equations.

Using PyFMI together with the Python 
simulation package `Assimulo <http://pypi.python.org/pypi/Assimulo>`_ 
adds industrial grade simulation capabilities of FMUs to Python.

Requirements:
-------------
- `FMI Library (at least 2.0.1) <https://github.com/modelon-community/fmi-library>`_
- `Numpy (recommended 1.6.2) <http://pypi.python.org/pypi/numpy>`_
- `Scipy (recommended 0.10.1) <http://pypi.python.org/pypi/scipy>`_
- `lxml (at least 2.3) <http://pypi.python.org/pypi/lxml>`_
- `Assimulo (at least 3.0) <http://pypi.python.org/pypi/Assimulo>`_
- `Cython (at least 0.18) <http://cython.org/>`_
- Python-headers (usually included on Windows, python-dev on Ubuntu)

Optional
---------
- `wxPython <http://pypi.python.org/pypi/wxPython>`_ For the Plot GUI.
- `matplotlib <http://pypi.python.org/pypi/matplotlib>`_ For the Plot GUI.

Source Installation:
----------------------

python setup.py install --fmil-home=/path/to/FMI_Library/

"""

copy_args=sys.argv[1:]

if O.getenv("FMIL_HOME"): #Check for if there exists and environment variable that specifies FMIL
    incdirs = O.path.join(O.getenv("FMIL_HOME"),'include')
    libdirs = O.path.join(O.getenv("FMIL_HOME"),'lib')
else:
    incdirs = ""
    libdirs = ""
    
static = False
debug_flag = False
fmilib_shared = ""
copy_gcc_lib = False
gcc_lib = None
force_32bit = False
no_msvcr = False
python3_flag = True if S.hexversion > 0x03000000 else False
with_openmp = False

static_link_gcc = "-static-libgcc"
flag_32bit = "-m32"
extra_c_flags = ""

# Fix path sep
for x in sys.argv[1:]:
    if not x.find('--prefix'):
        copy_args[copy_args.index(x)] = x.replace('/',O.sep)
    if not x.find('--fmil-home'):
        incdirs = O.path.join(x[12:],'include')
        libdirs = O.path.join(x[12:],'lib')
        copy_args.remove(x)
    if not x.find('--copy-libgcc'):
        if x[14:].upper() == "TRUE":
            copy_gcc_lib = True
        copy_args.remove(x)
    if not x.find('--static'):
        static = x[9:]
        if x[9:].upper() == "TRUE":
            static = True
        else:
            static = False
        copy_args.remove(x)
    if not x.find('--force-32bit'):
        if x[14:].upper() == "TRUE":
            force_32bit = True
        copy_args.remove(x)
    if not x.find('--no-msvcr'):
        if x[11:].upper() == "TRUE":
            no_msvcr = True
        copy_args.remove(x)
    if not x.find('--extra-c-flags'):
        extra_c_flags = x[16:]
        copy_args.remove(x)
    if not x.find('--with-openmp'):
        with_openmp = True
        copy_args.remove(x)
    if not x.find('--version'):
        VERSION = x[10:]
        copy_args.remove(x)
    if not x.find('--debug'):
        if x[8:].upper() == "TRUE":
            debug_flag = True
        else:
            debug_flag = False
        copy_args.remove(x)
    

if not incdirs:
    raise Exception("FMI Library cannot be found. Please specify its location, either using the flag to the setup script '--fmil-home' or specify it using the environment variable FMIL_HOME.")

#Check to see if FMILIB_SHARED exists and if so copy it
if 0 != sys.argv[1].find("clean"): #Dont check if we are cleaning!
    if sys.platform.startswith("win"):
        try:
            files = O.listdir(O.path.join(libdirs))
        except:
            raise Exception("The FMI Library binary cannot be found at path: "+str(O.path.join(libdirs)))
        for file in files:
            if "fmilib_shared" in file and not file.endswith("a"):
                shutil.copy2(O.path.join(libdirs,file),O.path.join(".","src","pyfmi"))
                fmilib_shared = O.path.join(".","src","pyfmi",file)
                break
        else:
            raise Exception("Could not find FMILibrary at: %s"%libdirs)
            
        if copy_gcc_lib:
            path_gcc_lib = ctypes.util.find_library("libgcc_s_dw2-1.dll")
            if path_gcc_lib != None:
                shutil.copy2(path_gcc_lib,O.path.join(".","src","pyfmi"))
                gcc_lib = O.path.join(".","src","pyfmi","libgcc_s_dw2-1.dll")

if no_msvcr:
    # prevent the MSVCR* being added to the DLLs passed to the linker
    def msvc_runtime_library_mod(): 
        return None
    
    import numpy.distutils
    numpy.distutils.misc_util.msvc_runtime_library = msvc_runtime_library_mod

def check_extensions():
    ext_list = []
    extra_link_flags = []
    
    if static:
        extra_link_flags.append(static_link_gcc)

    if force_32bit:
        extra_link_flags.append(flag_32bit)

    #COMMON PYX
    """
    ext_list = cythonize(["src"+O.path.sep+"common"+O.path.sep+"core.pyx"], 
                    include_path=[".","src","src"+O.sep+"common"],
                    include_dirs=[N.get_include()],pyrex_gdb=debug)
    
    ext_list[-1].include_dirs = [N.get_include(), "src","src"+O.sep+"common", incdirs]
        
    if debug:
        ext_list[-1].extra_compile_args = ["-g", "-fno-strict-aliasing", "-ggdb"]
        ext_list[-1].extra_link_args = extra_link_flags
    else:
        ext_list[-1].extra_compile_args = ["-O2", "-fno-strict-aliasing"]
        ext_list[-1].extra_link_args = extra_link_flags
    """
    
    #FMI PYX
    ext_list += cythonize(["src"+O.path.sep+"pyfmi"+O.path.sep+"fmi.pyx"], 
                    include_path=[".","src","src"+O.sep+"pyfmi"])
    
    #FMI UTIL
    ext_list += cythonize(["src"+O.path.sep+"pyfmi"+O.path.sep+"fmi_util.pyx"], 
                    include_path=[".","src","src"+O.sep+"pyfmi"])
    
    #FMI Extended PYX
    ext_list += cythonize(["src"+O.path.sep+"pyfmi"+O.path.sep+"fmi_extended.pyx"], 
                    include_path=[".","src","src"+O.sep+"pyfmi"])
                    
    #FMI Coupled PYX
    ext_list += cythonize(["src"+O.path.sep+"pyfmi"+O.path.sep+"fmi_coupled.pyx"], 
                    include_path=[".","src","src"+O.sep+"pyfmi"])
    
    #Simulation interface PYX
    ext_list += cythonize(["src"+O.path.sep+"pyfmi"+O.path.sep+"simulation"+O.path.sep+"assimulo_interface.pyx"], 
                    include_path=[".","src","src"+O.sep+"pyfmi"])
                    
    #MASTER PYX
    compile_time_env = {'WITH_OPENMP': with_openmp}
    ext_list += cythonize(["src"+O.path.sep+"pyfmi"+O.path.sep+"master.pyx"], 
                    include_path=[".","src","src"+O.sep+"pyfmi"], compile_time_env=compile_time_env)
    
    for i in range(len(ext_list)):
        
        ext_list[i].include_dirs = [N.get_include(), "src","src"+O.sep+"pyfmi", incdirs]
        ext_list[i].library_dirs = [libdirs]
        ext_list[i].language = "c"
        ext_list[i].libraries = ["fmilib_shared"] if sys.platform.startswith("win") else ["fmilib"] #If windows shared, else static
        
        if debug_flag:
            ext_list[i].extra_compile_args = ["-g", "-fno-strict-aliasing", "-ggdb"]
        else:
            ext_list[i].extra_compile_args = ["-O2", "-fno-strict-aliasing"]
        
        if force_32bit:
            ext_list[i].extra_compile_args.append(flag_32bit)
            
        if extra_c_flags:
            flags = extra_c_flags.split(' ')
            for f in flags:
                ext_list[i].extra_compile_args.append(f)
        
        ext_list[i].extra_link_args = extra_link_flags
        
        if with_openmp:
            ext_list[i].extra_link_args.append("-fopenmp")
            ext_list[i].extra_compile_args.append("-fopenmp")
        
        if python3_flag:
            ext_list[i].cython_directives = {"language_level": 3}

    return ext_list

ext_list = check_extensions()

try:
    from subprocess import Popen, PIPE
    _p = Popen(["svnversion", "."], stdout=PIPE)
    revision = _p.communicate()[0].decode('ascii')
except:
    revision = "unknown"
version_txt = 'src'+O.path.sep+'pyfmi'+O.path.sep+'version.txt'

#If a revision is found, always write it!
if revision != "unknown" and revision!="":
    with open(version_txt, 'w') as f:
        f.write(VERSION+'\n')
        f.write("r"+revision)
else:# If it does not, check if the file exists and if not, create the file!
    if not O.path.isfile(version_txt):
        with open(version_txt, 'w') as f:
            f.write(VERSION+'\n')
            f.write("unknown")
            
try:
    shutil.copy2('LICENSE', 'src'+O.path.sep+'pyfmi'+O.path.sep+'LICENSE')
    shutil.copy2('CHANGELOG', 'src'+O.path.sep+'pyfmi'+O.path.sep+'CHANGELOG')
except:
    pass

from numpy.distutils.core import setup
setup(name=NAME,
      version=VERSION,
      license=LICENSE,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      platforms=PLATFORMS,
      classifiers=CLASSIFIERS,
      ext_modules = ext_list,
      package_dir = {'pyfmi':'src'+O.path.sep+'pyfmi','pyfmi.common':'src'+O.path.sep+'common', 'pyfmi.tests':'tests'},
      packages=['pyfmi','pyfmi.simulation','pyfmi.examples','pyfmi.common','pyfmi.common.plotting', 'pyfmi.tests', 'pyfmi.common.log'],
      package_data = {'pyfmi':['examples'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'ME1.0'+O.path.sep+'*',
                               'examples'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'CS1.0'+O.path.sep+'*',
                               'examples'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'ME2.0'+O.path.sep+'*',
                               'examples'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'CS2.0'+O.path.sep+'*',
                               'tests'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'XML'+O.path.sep+'ME1.0'+O.path.sep+'*',
                               'tests'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'XML'+O.path.sep+'CS1.0'+O.path.sep+'*',
                               'tests'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'XML'+O.path.sep+'ME2.0'+O.path.sep+'*',
                               'tests'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'XML'+O.path.sep+'CS2.0'+O.path.sep+'*',
                               'tests'+O.path.sep+'files'+O.path.sep+'Results'+O.path.sep+'*',
                               'tests'+O.path.sep+'files'+O.path.sep+'Logs'+O.path.sep+'*',
                               'version.txt', 'LICENSE', 'CHANGELOG',
                               'util'+O.path.sep+'*']+(['*fmilib_shared*'] if sys.platform.startswith("win") else [])+(['libgcc_s_dw2-1.dll'] if copy_gcc_lib else [])},
      script_args=copy_args
      )


#Dont forget to delete fmilib_shared
if 0 != sys.argv[1].find("clean"): #Dont check if we are cleaning!
    if sys.platform.startswith("win"):
        if O.path.exists(fmilib_shared):
            O.remove(fmilib_shared)
        if gcc_lib and O.path.exists(gcc_lib):
            O.remove(gcc_lib)
