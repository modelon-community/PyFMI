###########
Tutorial
###########

This tutorial is intended to give a short introduction on how to use the PyFMI package to load an FMU into Python and to simulate the given model.

For a more detailed description on how to use PyFMI, see the user's documentation in `JModelica.org <http://www.jmodelica.org/page/236>`_

Loading an FMU into Python
============================

Loading of an FMU is performed by simply importing the necessary object (*FMUModel*) from PyFMI. The object takes care of unzipping the model, loading the XML description and connecting the binary for use from Python.

.. code-block:: python

    # Import the model object (FMUModel)
    from pyfmi import FMUModel
    
    #Load the FMU
    model = FMUModel('myFMU.fmu')


Simulating an FMU
========================

Simulation of an FMU requires that the additional package `Assimulo <http://www.jmodelica.org/assimulo>`_ is available and is performed simply by using the *simulate* method:

.. code-block:: python

    #Simulate an FMU
    res = model.simulate(final_time=10)

This will simulate the model from its default starting time 0.0 to 10.0 by using default options. The result is returned and stored in *res*.

Information about the arguments to *simulate* is best viewed interactively from for example an IPython shell:

.. code-block:: python

    #View the documentation for simulate
    model.simulate?


Options
------------

The options for an algorithm, which in our case is Assimulo, can be retrieved by calling the method *simulate_options*:  

.. code-block:: python
    
    #Get the default options
    opts = model.simulate_options()

This will return the default options for a simulation as a dictionary, for example the default solver is *CVode* from the Assimulo package. Changing the options specifically related to the solver is done by:

.. code-block:: python

    opts["CVode_options"]["atol"] = 1e-6 #Change the absolute tolerance
    opts["CVode_options"]["discr"] = "Adams" #Change the discretization from BDF to Adams
    
All options related to the *CVode* solver can be found in the Assimulo documentation. 

For changing general options such as the number of output points (number of communication points, ncp), they are accessed directly:

.. code-block:: python

    opts["ncp"] = 100 #One hundred output points

To use the options in an simulation, pass them in the call to the *simulate* method:

.. code-block:: python

    res = model.simulate(final_time=10, options=opts)


Currently the only solver that supports fully the `FMI <http://www.modelisar.com>`_ specification is *CVode*.



Result Object
---------------

The result object returned from a simulation contains all trajectories related to the variables in the model and are accessed as a dictionary.

.. code-block:: python

    res = model.simulate()

    y = res['y'] #Return the result for the variable/parameter/constant y
    dery = res['der(y)'] #Return the result for the variable/parameter/constant der(y)

This can be done for all the variables, parameters and constants defined in the model and is the preferred way of retrieving the result.




Additional information
========================

The PyFMI package comes with a number of examples, showing how to simulate different problems. These examples can be found :doc:`here <examples>`. 
