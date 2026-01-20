###########
Tutorial
###########

This tutorial is intended to give a short introduction on how to use the PyFMI package to load an FMU into Python and to simulate the given model.

Loading an FMU into Python
============================

Loading of an FMU is performed by simply importing the necessary object (*load_fmu*) from PyFMI. The object takes care of unzipping the model, loading the XML description and connecting the binary for use from Python.

.. code-block:: python

    # Import the load function (load_fmu)
    from pyfmi import load_fmu
    
    #Load the FMU
    model = load_fmu('myFMU.fmu')

Note that this will either return an instance of a class consistent with the FMI for Model Exchange definition or for the Co-Simulation definition (and of the correct FMI version).

Simulating an FMU
========================

Simulation of an FMU exported as Model Exchange requires that the additional package `Assimulo <http://www.jmodelica.org/assimulo>`_ is available. For Co-Simulation FMUs, no additional package is required as the solver is contained inside the FMU. A simulation is performed simply by using the *simulate* method:

.. code-block:: python

    #Simulate an FMU
    res = model.simulate(final_time=10)

This will simulate the model from its default starting time 0.0 to 10.0 using default options. The result is returned and stored in *res*.

Information about the arguments to *simulate* is best viewed interactively from for example an IPython shell:

.. code-block:: python

    #View the documentation for simulate
    model.simulate?

Options (for Model Exchange)
------------------------------

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

Options (for Co-Simulation)
-----------------------------

The simulation options for a Co-Simulation FMU is retrieved and set consistent as for a Model Exchange FMU. The only difference is the actual options.


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
