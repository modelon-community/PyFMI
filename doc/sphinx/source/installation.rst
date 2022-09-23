.. highlight:: rest

=============
Installation
=============


Current version:

    Available on PyPI, http://pypi.python.org/pypi/PyFMI

Additionally, PyFMI is available through `conda <http://conda.pydata.org/docs/index.html>`_::

    conda install -c conda-forge pyfmi

Requirements:
-------------
- `FMI Library <http://www.jmodelica.org/FMILibrary>`_
- `Numpy <http://pypi.python.org/pypi/numpy>`_
- `Scipy <http://pypi.python.org/pypi/scipy>`_
- `lxml <http://pypi.python.org/pypi/lxml>`_
- `Assimulo <http://pypi.python.org/pypi/Assimulo>`_

Optional
---------
- `wxPython <http://pypi.python.org/pypi/wxPython>`_ For the Plot GUI.
- `matplotlib <http://pypi.python.org/pypi/matplotlib>`_ For the Plot GUI.


Once all the requirements are satisfied the installation is performed using the command::

    python setup.py install --fmil-home=/path/to/fmil
    
Where the flag "--fmil-home" should point to where FMI Library has been installed.
