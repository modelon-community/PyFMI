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

__all__ = ['fmi_algorithm_drivers', 'examples', 'fmi', 'common']

#Import the model class allowing for users to type e.g.,: from pyfmi import FMUModelME1
from pyfmi.fmi import load_fmu
from pyfmi.fmi1 import FMUModelME1, FMUModelCS1
from pyfmi.fmi2 import FMUModelME2, FMUModelCS2
from pyfmi.fmi_coupled import CoupledFMUModelME2
from pyfmi.master import Master
from pyfmi.fmi_extended import FMUModelME1Extended
import os.path
import sys
import time

try:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    _fpath=os.path.join(curr_dir,'version.txt')
    with open(_fpath, 'r') as f:
        __version__=f.readline().strip()
        __revision__=f.readline().strip()
except Exception:
    __version__ = "unknown"
    __revision__= "unknown"


def check_packages():
    le=30
    le_short=15
    startstr = "Performing pyfmi package check"
    sys.stdout.write("\n")
    sys.stdout.write(startstr+" \n")
    sys.stdout.write("="*len(startstr))
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    time.sleep(0.25)
    
    # print pyfmi version
    sys.stdout.write(
        "%s %s" %("PyFMI version ".ljust(le,'.'),(__version__).ljust(le)+"\n\n"))
    sys.stdout.flush()
    time.sleep(0.25)
    
    # check os
    platform = sys.platform
    sys.stdout.write(
        "%s %s" %("Platform ".ljust(le,'.'),(str(platform)).ljust(le)+"\n\n"))
    sys.stdout.flush()
    time.sleep(0.25)
    
    #check python version
    pyversion = sys.version.partition(" ")[0]
    sys.stdout.write(
        "%s %s" % ("Python version ".ljust(le,'.'),pyversion.ljust(le)))
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    time.sleep(0.25)
    
    import importlib
    # Test dependencies
    sys.stdout.write("\n\n")
    sys.stdout.write("Dependencies: \n\n".rjust(0))
    modstr="Package"
    verstr="Version"
    sys.stdout.write("%s %s" % (modstr.ljust(le), verstr.ljust(le)))
    sys.stdout.write("\n")
    sys.stdout.write(
        "%s %s" % (("-"*len(modstr)).ljust(le), ("-"*len(verstr)).ljust(le)))
    sys.stdout.write("\n")
    
    packages=["assimulo", "Cython", "matplotlib", "numpy", "scipy", "wxPython"]
    
    if platform == "win32":
        packages.append("pyreadline")
        packages.append("setuptools")
    
    error_packages=[]
    warning_packages=[]
    fp = None
    for package in packages:
        try:
            vers="--"
            mod = importlib.import_module(package)

            try:
                if package == "pyreadline":
                    vers = mod.release.version
                else:
                    vers = mod.__version__
            except AttributeError:
                pass
            sys.stdout.write("%s %s" %(package.ljust(le,'.'), vers.ljust(le)))
        except ImportError:
            if package == "assimulo" or package == "wxPython":
                sys.stdout.write("%s %s %s" % (package.ljust(le,'.'), vers.ljust(le_short), "Package missing - Warning issued, see details below".ljust(le_short)))
                warning_packages.append(package)
            else:
                sys.stdout.write("%s %s %s " % (package.ljust(le,'.'), vers.ljust(le_short), "Package missing - Error issued, see details below.".ljust(le_short)))
                error_packages.append(package)
            pass
        finally:
            if fp:
                fp.close()
        sys.stdout.write("\n")
        sys.stdout.flush()
        time.sleep(0.25)

        
    # Write errors and warnings
    # are there any errors?
    if len(error_packages) > 0:
        sys.stdout.write("\n")
        errtitle = "Errors"
        sys.stdout.write("\n")
        sys.stdout.write(errtitle+" \n")
        sys.stdout.write("-"*len(errtitle))
        sys.stdout.write("\n\n")
        sys.stdout.write("The package(s): \n\n")
        
        for er in error_packages:
            sys.stdout.write("   - "+str(er))
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.write("could not be found. It is not possible to run the pyfmi \
package without it/them.\n")
    
    if len(warning_packages) > 0:
        sys.stdout.write("\n")
        wartitle = "Warnings"
        sys.stdout.write("\n")
        sys.stdout.write(wartitle+" \n")
        sys.stdout.write("-"*len(wartitle))
        sys.stdout.write("\n\n")
        
        for w in warning_packages:
            if w == 'assimulo':
                sys.stdout.write("-- The package assimulo could not be found. \
This package is needed to be able to simulate FMUs. Also, some of the examples \
in pyfmi.examples will not work.")
            elif w == 'wxPython':
                sys.stdout.write("-- The package wxPython could not be found. \
This package is needed to be able to use the plot-GUI.")

            sys.stdout.write("\n\n")
