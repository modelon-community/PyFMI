About
-----------

PyFMI is a package for loading and interacting with Functional Mock-Up Units
(FMUs) both for Model Exchange and Co-Simulation, which are compiled dynamic
models compliant with the Functional Mock-Up Interface ([FMI](https://fmi-standard.org/)).

For a more indebt technical description of the features / functionality see the following [link](https://portal.research.lu.se/portal/files/7201641/pyfmi_tech.pdf).

For citing PyFMI, please use:

<em>Andersson, C, Åkesson, J & Führer, C 2016, PyFMI: A Python Package for Simulation of Coupled Dynamic Models with the Functional Mock-up Interface. Technical Report in Mathematical Sciences, nr. 2, vol. 2016, vol. LUTFNA-5008-2016, Centre for Mathematical Sciences, Lund University</em>.

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

FMU Import Compliance
-----------
PyFMI is tested daily in several different ways:
* Extensive external testing within the `Optimica Compiler Toolkit` (OCT) provided by [Modelon](https://help.modelon.com/latest/reference/oct/) using all models in Modelon Modelica libararies (35000+ models), exported as FMUs.
* Unit tests from the `tests` directory. These use FMUs generated by
* * OCT
  * JModelica.org
  * FMUs in the example directory are generated via <em>FMU SDK by Qtronic</em>, more information is available in related `README.txt` for these FMUs with proper license text. 
* PyFMI is the default execution engine for [Modelon Impact](https://modelon.com/modelon-impact/).
* Some testing with Dymola FMUs and FMUs from [FMI Cross Check](https://github.com/modelica/fmi-cross-check).
