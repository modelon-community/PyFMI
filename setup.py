#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2014 Modelon AB
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
from distutils.core import setup

import distutils
import os as O
import sys as S
import shutil
import numpy as N
import ctypes.util
import sys

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    raise Exception("Please upgrade to a newer Cython version, >= 0.15.")

NAME = "PyFMI"
AUTHOR = "Modelon AB"
AUTHOR_EMAIL = ""
VERSION = "trunk"
LICENSE = "LGPL"
URL = "http://www.jmodelica.org"
DOWNLOAD_URL = "http://www.jmodelica.org/page/12"
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
is based on FMI Library, see http://www.jmodelica.org/FMILibrary .

FMI is a standard that enables tool independent exchange of dynamic 
models on binary format. Several industrial simulation platforms 
supports export of FMUs, including, Dymola, JModelica.org, OpenModelica 
and SimulationX, see https://www.fmi-standard.org/tools 
for a complete list. PyFMI offers a Python interface for interacting 
with FMUs and enables for example loading of FMU models, setting of 
model parameters and evaluation of model equations.

PyFMI is available as a stand-alone package or as part of the 
JModelica.org distribution. Using PyFMI together with the Python 
simulation package `Assimulo <http://pypi.python.org/pypi/Assimulo>`_ adds industrial grade simulation 
capabilities of FMUs to Python.

For a forum discussing usage and development of PyFMI, see http://www.jmodelica.org/forum.

Requirements:
-------------
- `FMI Library (at least 2.0.1) <http://www.jmodelica.org/FMILibrary>`_
- `Numpy (recommended 1.6.2) <http://pypi.python.org/pypi/numpy>`_
- `Scipy (recommended 0.10.1) <http://pypi.python.org/pypi/scipy>`_
- `lxml (at least 2.3) <http://pypi.python.org/pypi/lxml>`_
- `Assimulo (at least 2.6) <http://pypi.python.org/pypi/Assimulo>`_
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

####NECESSECARY FOR THE DEPRECATED FMI LOGGER
#Load the helper function
if sys.platform == 'win32':
    suffix = '.dll'
elif sys.platform == 'darwin':
    suffix = '.dylib'
else:
    suffix = '.so'

path_log_src = "src"+O.path.sep+"pyfmi"+O.path.sep+"util" + O.path.sep + "FMILogger.c"
path_log_dest = "src"+O.path.sep+"pyfmi"+O.path.sep+"util" + O.path.sep + "FMILogger" + suffix

if force_32bit:
    O.system("gcc "+flag_32bit+" -fPIC " + extra_c_flags +' '+ path_log_src+" -shared -o"+path_log_dest)
else:
    O.system("gcc -fPIC "+ extra_c_flags + ' ' + path_log_src+" -shared -o"+path_log_dest)

########


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
        f.write("r"+revision)
else:# If it does not, check if the file exists and if not, create the file!
    if not O.path.isfile(version_txt):
        with open(version_txt, 'w') as f:
            f.write("unknown")

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
      package_dir = {'pyfmi':'src'+O.path.sep+'pyfmi','pyfmi.common':'src'+O.path.sep+'common'},
      packages=['pyfmi','pyfmi.simulation','pyfmi.examples','pyfmi.common','pyfmi.common.plotting'],
      package_data = {'pyfmi':['examples'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'ME1.0'+O.path.sep+'*',
                               'examples'+O.path.sep+'files'+O.path.sep+'FMUs'+O.path.sep+'CS1.0'+O.path.sep+'*',
                               'version.txt',
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
