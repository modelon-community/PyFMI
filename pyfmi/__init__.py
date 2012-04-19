#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2010 Modelon AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
The JModelica.org Python package for working with FMI <http:/www.jmodelica.org/>
"""

__all__ = ['fmi_algorithm_drivers', 'examples', 'fmi', 'common']

#Import the model class allowing for users to type: from pyfmi import FMUModel
from fmi import FMUModel
import numpy as N

int = N.int32
N.int = N.int32


def check_packages():
    import sys, time
    le=30
    startstr = "Performing pyfmi package check"
    sys.stdout.write("\n")
    sys.stdout.write(startstr+" \n")
    sys.stdout.write("="*len(startstr))
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    time.sleep(0.25)

    # check os
    platform = sys.platform
    sys.stdout.write(
        "%s %s" %("Platform".ljust(le,'.'),(str(platform)).ljust(le)+"\n\n"))
    sys.stdout.flush()
    time.sleep(0.25)
    
    #check python version
    pyversion = sys.version.partition(" ")[0]
    sys.stdout.write(
        "%s %s" % ("Python version:".ljust(le,'.'),pyversion.ljust(le)))
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    time.sleep(0.25)
    
    import imp
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
    
    packages=["numpy", "scipy", "matplotlib", "lxml", "assimulo", "wxPython"]
    
    if platform == "win32":
        packages.append("pyreadline")
        packages.append("setuptools")
    
    error_packages=[]
    warning_packages=[]
    fp = None
    for package in packages:
        try:
            vers="n/a"
            fp, path, desc = imp.find_module(package)
            mod = imp.load_module(package, fp, path, desc)

            try:
                if package == "pyreadline":
                    vers = mod.release.version
                elif package == "lxml":
                    from lxml import etree
                    vers = etree.__version__
                else:
                    vers = mod.__version__
            except AttributeError, e:
                pass
            sys.stdout.write("%s %s %s" %(package.ljust(le,'.'), vers.ljust(le), "Ok".ljust(le)))
        except ImportError, e:
            if package == "assimulo" or package == "wxPython":
                sys.stdout.write("%s %s %s" % (package.ljust(le,'.'), vers.ljust(le), "Package missing - Warning issued, see details below".ljust(le)))
                warning_packages.append(package)
            else:
                sys.stdout.write("%s %s %s " % (package.ljust(le,'.'), vers.ljust(le), "Package missing - Error issued, see details below.".ljust(le)))
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
        sys.stdout.write("The packages: \n\n")
        
        for er in error_packages:
            sys.stdout.write("   - "+str(er))
            sys.stdout.write("\n")
        sys.stdout.write("\n")
        sys.stdout.write("could not be found. It is not possible to run \
        the pyfmi package without them.\n")
    
    if len(warning_packages) > 0:
        sys.stdout.write("\n")
        wartitle = "Warnings"
        sys.stdout.write("\n")
        sys.stdout.write(wartitle+" \n")
        sys.stdout.write("-"*len(wartitle))
        sys.stdout.write("\n\n")
        
        for w in warning_packages:
            if w == 'assimulo':
                sys.stdout.write("** The package assimulo could not be found. \n  \
 This package is needed to be able to use: \n\n   \
- pyfmi.FMUModel.simulate with default argument \"algorithm\" = AssimuloAlg \n   \
- The pyfmi.simulation package \n   \
- Some of the examples in the pyfmi.examples package")
            elif w == 'wxPython':
                sys.stdout.write("** The package wxPython could not be found.\n \
This package is needed to be able to use the plot-GUI.")

            sys.stdout.write("\n\n")


