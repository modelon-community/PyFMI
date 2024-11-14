#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2014-2024 Modelon AB
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

import os
import shutil
import numpy as np
import ctypes.util
import sys

try:
    from numpy.distutils.core import setup
    have_nd = True
except ImportError:
    from setuptools import setup
    have_nd = False

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    raise Exception("Please upgrade to a newer Cython version, >= 3.")


NAME = "PyFMI"
AUTHOR = "Modelon AB"
AUTHOR_EMAIL = ""
VERSION = "2.15.0"
LICENSE = "LGPL"
URL = "https://github.com/modelon-community/PyFMI"
DOWNLOAD_URL = "https://github.com/modelon-community/PyFMI/releases"
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
- `Python-headers (usually included on Windows, python-dev on Ubuntu)`_
- `Python 3.9 or newer`_
- Python package dependencies are listed in file setup.cfg.

Optional
---------
- `wxPython <http://pypi.python.org/pypi/wxPython>`_ For the Plot GUI.
- `matplotlib <http://pypi.python.org/pypi/matplotlib>`_ For the Plot GUI.

Source Installation (note that assimulo needs to be installed and on PYTHONPATH in order to install pyfmi):
----------------------

python setup.py install --fmil-home=/path/to/FMI_Library/

"""

copy_args = sys.argv[1:]

fmil_home = os.getenv("FMIL_HOME")
if fmil_home: #Check for environment variable that specifies FMIL
    incdirs = os.path.join(fmil_home, 'include')
    libdirs = os.path.join(fmil_home, 'lib')
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
with_openmp = False

static_link_gcc = "-static-libgcc"
flag_32bit = "-m32"
extra_c_flags = ""

# Fix path sep
for x in sys.argv[1:]:
    if not x.find('--prefix'):
        if not have_nd:
            raise Exception("Cannot specify --prefix without numpy.distutils")
        copy_args[copy_args.index(x)] = x.replace('/',os.sep)
    if not x.find('--fmil-home'):
        incdirs = os.path.join(x[12:],'include')
        libdirs = os.path.join(x[12:],'lib')
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
            if not have_nd:
                raise Exception("Cannot specify --no-msvcr without numpy.distutils")
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
            files = os.listdir(os.path.join(libdirs))
        except Exception:
            raise Exception("The FMI Library binary cannot be found at path: "+str(os.path.join(libdirs)))
        for file in files:
            if "fmilib_shared" in file and not file.endswith("a"):
                shutil.copy2(os.path.join(libdirs,file),os.path.join(".","src","pyfmi"))
                fmilib_shared = os.path.join(".","src","pyfmi",file)
                break
        else:
            raise Exception("Could not find FMILibrary at: %s"%libdirs)
            
        if copy_gcc_lib:
            path_gcc_lib = ctypes.util.find_library("libgcc_s_dw2-1.dll")
            if path_gcc_lib is not None:
                shutil.copy2(path_gcc_lib,os.path.join(".","src","pyfmi"))
                gcc_lib = os.path.join(".","src","pyfmi","libgcc_s_dw2-1.dll")

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
    ext_list = cythonize([os.path.join("src", "common", "core.pyx")], 
                    include_path=[".","src",os.path.join("src", "common")],
                    include_dirs=[N.get_include()],pyrex_gdb=debug)
    
    ext_list[-1].include_dirs = [N.get_include(), "src",os.path.join("src", "common"), incdirs]
        
    if debug:
        ext_list[-1].extra_compile_args = ["-g", "-fno-strict-aliasing", "-ggdb"]
        ext_list[-1].extra_link_args = extra_link_flags
    else:
        ext_list[-1].extra_compile_args = ["-O2", "-fno-strict-aliasing"]
        ext_list[-1].extra_link_args = extra_link_flags
    """
    incl_path = [".", "src", os.path.join("src", "pyfmi")]
    #FMI PYX
    ext_list += cythonize([os.path.join("src", "pyfmi", "fmi.pyx")], 
                    include_path = incl_path,
                    compiler_directives={'language_level' : "3str"})
    
    #FMI UTIL
    ext_list += cythonize([os.path.join("src", "pyfmi", "fmi_util.pyx")], 
                    include_path = incl_path,
                    compiler_directives={'language_level' : "3str"})
    
    #FMI Extended PYX
    ext_list += cythonize([os.path.join("src", "pyfmi", "fmi_extended.pyx")], 
                    include_path = incl_path,
                    compiler_directives={'language_level' : "3str"})
                    
    #FMI Coupled PYX
    ext_list += cythonize([os.path.join("src", "pyfmi", "fmi_coupled.pyx")], 
                    include_path = incl_path,
                    compiler_directives={'language_level' : "3str"})
    
    #Simulation interface PYX
    ext_list += cythonize([os.path.join("src", "pyfmi", "simulation", "assimulo_interface.pyx")], 
                    include_path = incl_path,
                    compiler_directives={'language_level' : "3str"})
                    
    #MASTER PYX
    compile_time_env = {'WITH_OPENMP': with_openmp}
    ext_list += cythonize([os.path.join("src", "pyfmi", "master.pyx")], 
                    include_path = incl_path, 
                    compile_time_env=compile_time_env,
                    compiler_directives={'language_level' : "3str"})
    
    # Test utilities
    ext_list += cythonize([os.path.join("src", "pyfmi", "tests", "test_util.pyx")], 
                    include_path = incl_path, 
                    compiler_directives={'language_level' : "3str"})
    
    for i in range(len(ext_list)):
        
        ext_list[i].include_dirs = [np.get_include(), "src", os.path.join("src", "pyfmi"), incdirs]
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
        
        ext_list[i].cython_directives = {"language_level": 3}

    return ext_list

ext_list = check_extensions()

try:
    from subprocess import Popen, PIPE
    _p = Popen(["svnversion", "."], stdout=PIPE)
    revision = _p.communicate()[0].decode('ascii')
except Exception:
    revision = "unknown"
version_txt = os.path.join('src', 'pyfmi', 'version.txt')

#If a revision is found, always write it!
if revision != "unknown" and revision!="":
    with open(version_txt, 'w') as f:
        f.write(VERSION+'\n')
        f.write("r"+revision)
else:# If it does not, check if the file exists and if not, create the file!
    if not os.path.isfile(version_txt):
        with open(version_txt, 'w') as f:
            f.write(VERSION+'\n')
            f.write("unknown")
            
try:
    shutil.copy2('LICENSE', os.path.join('src', 'pyfmi', 'LICENSE'))
    shutil.copy2('CHANGELOG', os.path.join('src', 'pyfmi', 'CHANGELOG'))
except Exception:
    pass
extra_package_data = ['*fmilib_shared*'] if sys.platform.startswith("win") else []
extra_package_data += ['libgcc_s_dw2-1.dll'] if copy_gcc_lib else []

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
      package_dir = {'pyfmi':        os.path.join('src', 'pyfmi'),
                     'pyfmi.common': os.path.join('src', 'common'),
                     'pyfmi.tests':  'tests'},
      packages=[
        'pyfmi',
        'pyfmi.simulation',
        'pyfmi.examples',
        'pyfmi.common',
        'pyfmi.common.plotting',
        'pyfmi.tests',
        'pyfmi.common.log'
      ],
      package_data = {'pyfmi': [
        'examples/files/FMUs/ME1.0/*',
        'examples/files/FMUs/CS1.0/*',
        'examples/files/FMUs/ME2.0/*',
        'examples/files/FMUs/CS2.0/*',
        'tests/files/FMUs/XML/ME1.0/*',
        'tests/files/FMUs/XML/CS1.0/*',
        'tests/files/FMUs/XML/ME2.0/*',
        'tests/files/FMUs/XML/CS2.0/*',
        'tests/files/Results/*',
        'tests/files/Logs/*',
        'version.txt',
        'LICENSE',
        'CHANGELOG',
        'util/*'] + extra_package_data},
      script_args=copy_args
      )


#Dont forget to delete fmilib_shared
if 0 != sys.argv[1].find("clean"): #Dont check if we are cleaning!
    if sys.platform.startswith("win"):
        if os.path.exists(fmilib_shared):
            os.remove(fmilib_shared)
        if gcc_lib and os.path.exists(gcc_lib):
            os.remove(gcc_lib)
