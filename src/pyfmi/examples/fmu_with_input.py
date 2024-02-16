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

import os
import numpy as np
import pylab as pl

from pyfmi import load_fmu

curr_dir = os.path.dirname(os.path.abspath(__file__))
path_to_fmus = os.path.join(curr_dir, 'files', 'FMUs')
path_to_fmus_me1 = os.path.join(path_to_fmus,"ME1.0")
path_to_fmus_cs1 = os.path.join(path_to_fmus,"CS1.0")

def run_demo(with_plots=True):
    """
    Demonstrates how to simulate an FMU with inputs.
    
    See also simulation_with_input.py
    """
    fmu_name = os.path.join(path_to_fmus_me1, 'SecondOrder.fmu')

    # Generate input
    t = np.linspace(0.,10.,100) 
    u = np.cos(t)
    u_traj = np.transpose(np.vstack((t,u)))
    
    # Create input object
    input_object = ('u', u_traj)

    # Load the dynamic library and XML data
    model = load_fmu(fmu_name)
    
    # Set the first input value to the model
    model.set('u',u[0])
    
    # Simulate
    res = model.simulate(final_time=30, input=input_object, options={'ncp':3000})

    x1_sim = res['x1']
    x2_sim = res['x2']
    u_sim = res['u']
    time_sim = res['time']
        
    assert np.abs(res.final('x1')*1.e1 - (-8.3999640)) < 1e-3
    assert np.abs(res.final('x2')*1.e1 - (-5.0691179)) < 1e-3  
    assert np.abs(res.final('u')*1.e1 - (-8.3907153)) < 1e-3 

    if with_plots:
        pl.figure()
        pl.subplot(2,1,1)
        pl.plot(time_sim, x1_sim, time_sim, x2_sim)
        pl.subplot(2,1,2)
        pl.plot(time_sim, u_sim,'x-',t, u[:],'x-')
        pl.show()

if __name__=="__main__":
    run_demo()
