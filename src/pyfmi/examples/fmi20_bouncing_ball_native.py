#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2018 Modelon AB
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
import pylab as P
import numpy as N

from pyfmi import load_fmu

curr_dir = os.path.dirname(os.path.abspath(__file__))
path_to_fmus = os.path.join(curr_dir, 'files', 'FMUs')
path_to_fmus_me2 = os.path.join(path_to_fmus,"ME2.0")

def run_demo(with_plots=True):
    """
    This example shows how to use the raw (JModelica.org) FMI interface for
    simulation of an FMU.
    
    FMU = bouncingBall.fmu 
    (Generated using Qtronic FMU SDK (http://www.qtronic.de/en/fmusdk.html) )
    
    This example is written similar to the example in the documentation of the 
    'Functional Mock-up Interface for Model Exchange' version 2.0 
    (http://www.functional-mockup-interface.org/) 
    """
    
    #Load the FMU by specifying the fmu and the directory
    bouncing_fmu = load_fmu(os.path.join(path_to_fmus_me2, 'bouncingBall.fmu'))

    Tstart = 0.5 #The start time.
    Tend   = 3.0 #The final simulation time.
    rtol   = 1e-6 ## relative tolerance
    
    # Initialize the model. Also sets all the start attributes defined in the 
    # XML file.
    bouncing_fmu.setup_experiment(start_time = Tstart) 
    bouncing_fmu.enter_initialization_mode()
    bouncing_fmu.exit_initialization_mode()

    eInfo = bouncing_fmu.get_event_info()
    eInfo.newDiscreteStatesNeeded = True
    #Event iteration
    while eInfo.newDiscreteStatesNeeded:
        bouncing_fmu.enter_event_mode()
        bouncing_fmu.event_update()
        eInfo = bouncing_fmu.get_event_info()
    
    bouncing_fmu.enter_continuous_time_mode()
    #Get Continuous States
    x = bouncing_fmu.continuous_states
    #Get the Nominal Values
    x_nominal = bouncing_fmu.nominal_continuous_states
    #Get the Event Indicators
    event_ind = bouncing_fmu.get_event_indicators()
    
    #For retrieving the solutions use,
    #bouncing_fmu.get_real,get_integer,get_boolean,get_string (valueref)
    
    #Values for the solution
    #Retrieve the valureferences for the values 'h' and 'v'
    vref  = [bouncing_fmu.get_variable_valueref('h')] + [bouncing_fmu.get_variable_valueref('v')] 

    t_sol = [Tstart]
    sol = [bouncing_fmu.get_real(vref)]
    
    #Main integration loop.
    time = Tstart
    Tnext = Tend #Used for time events
    dt = 0.01 #Step-size
    
    while time < Tend and not bouncing_fmu.get_event_info().terminateSimulation:
        #Compute the derivative
        dx = bouncing_fmu.get_derivatives()
        
        #Advance
        h = min(dt, Tnext-time)
        time = time + h

        #Set the time
        bouncing_fmu.time = time
        
        #Set the inputs at the current time (if any)
        #bouncing_fmu.set_real,set_integer,set_boolean,set_string (valueref, values)
        
        #Set the states at t = time (Perform the step)
        x = x + h*dx
        bouncing_fmu.continuous_states = x
        
        #Get the event indicators at t = time
        event_ind_new = bouncing_fmu.get_event_indicators()
        
        #Inform the model about an accepted step and check for step events
        step_event, terminate = bouncing_fmu.completed_integrator_step()
        if terminate: 
            return
        
        #Check for time and state events
        time_event  = abs(time-Tnext) <= 1.e-10
        state_event = True in ((event_ind_new>0.0) != (event_ind>0.0))

        #Event handling
        if step_event or time_event or state_event:
            bouncing_fmu.enter_event_mode()
            eInfo = bouncing_fmu.get_event_info()
            eInfo.newDiscreteStatesNeeded = True
            #Event iteration
            while eInfo.newDiscreteStatesNeeded:
                bouncing_fmu.event_update(intermediateResult=True) #Stops after each event iteration
                eInfo = bouncing_fmu.get_event_info()

                #Retrieve solutions (if needed)
                if eInfo.newDiscreteStatesNeeded:
                    #bouncing_fmu.get_real, get_integer, get_boolean, 
                    # get_string(valueref)
                    pass
            
            #Check if the event affected the state values and if so sets them
            if eInfo.valuesOfContinuousStatesChanged:
                x = bouncing_fmu.continuous_states
        
            #Get new nominal values.
            if eInfo.nominalsOfContinuousStatesChanged:
                atol = 0.01*rtol*bouncing_fmu.nominal_continuous_states
                
            #Check for new time event
            if eInfo.nextEventTimeDefined:
                Tnext = min(eInfo.nextEventTime, Tend)
            else:
                Tnext = Tend
            bouncing_fmu.enter_continuous_time_mode()
        
        event_ind = event_ind_new
        
        #Retrieve solutions at t=time for outputs
        #bouncing_fmu.get_real,get_integer,get_boolean,get_string (valueref)
        
        t_sol += [time]
        sol += [bouncing_fmu.get_real(vref)]
    
    
    #Plot the solution
    if with_plots:
        #Plot the height
        P.figure(1)
        P.plot(t_sol,N.array(sol)[:,0])
        P.title(bouncing_fmu.get_name())
        P.ylabel('Height (m)')
        P.xlabel('Time (s)')
        #Plot the velocity
        P.figure(2)
        P.plot(t_sol,N.array(sol)[:,1])
        P.title(bouncing_fmu.get_name())
        P.ylabel('Velocity (m/s)')
        P.xlabel('Time (s)')
        P.show()
        
if __name__ == "__main__":
    run_demo()
