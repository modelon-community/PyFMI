PyFMI
=====

PyFMI is a package for loading and interacting with Functional Mock-Up Units
(FMUs) both for Model Exchange and Co-Simulation, which are compiled dynamic
models compliant with the Functional Mock-Up Interface (FMI). See
 https://fmi-standard.org/ for more
information.

For a more indebt technical description of the features / functionality see:
https://portal.research.lu.se/portal/files/7201641/pyfmi_tech.pdf

For citing PyFMI, please use:

Andersson, C, Åkesson, J & Führer, C 2016, PyFMI: A Python Package for Simulation of Coupled Dynamic Models with the Functional Mock-up Interface. Technical Report in Mathematical Sciences, nr. 2, vol. 2016, vol. LUTFNA-5008-2016, Centre for Mathematical Sciences, Lund University.

For information about contributing, see [here](https://github.com/modelon/contributing).

Installation from source
-----------
PyFMI can be built and installed from source, note this requires [FMI library](https://github.com/modelon-community/fmi-library). Additional details are found in the file `setup.py`.

`python setup.py install --fmil-home=/path/to/FMI_Library/`

Here the flag "--fmil-home" points to where FMI Library is installed. The Python package [Assimulo](https://github.com/modelon-community/Assimulo) needs to be installed in order to build from source.
Installation using CONDA
-----------

`conda install -c conda-forge pyfmi`

Note that some examples requires optional dependencies for plotting, they are:
- wxPython
- matplotlib
