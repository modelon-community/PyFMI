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

import os as O
import numpy as N
import pylab as p

from pyfmi import FMUModel

curr_dir = O.path.dirname(O.path.abspath(__file__));
path_to_fmus = O.path.join(curr_dir, 'files', 'FMUs')

def run_demo(with_plots=True):
    """
    Demonstrates how to simulate an FMU with an input function.
    
    See also simulation_with_input.py
    """
    fmu_name = O.path.join(path_to_fmus,'SecondOrder.fmu')

    # Create input object
    input_object = ('u', N.cos)

    # Load the dynamic library and XML data
    model=FMUModel(fmu_name)

    # Simulate
    res = model.simulate(final_time=30, input=input_object, options={'ncp':3000})
    
    x1_sim = res['x1']
    x2_sim = res['x2']
    u_sim = res['u']
    time_sim = res['time']
    
    assert N.abs(x1_sim[-1] - (-1.646485144)) < 1e-3, \
            "Wrong value of x1_sim function in simulation_with_input.py"

    assert N.abs(x2_sim[-1]*1.e1 - (-7.30591626709)) < 1e-3, \
            "Wrong value of x2_sim function in simulation_with_input.py"  

    assert N.abs(u_sim[-1]*1.e1 - (1.54251449888)) < 1e-3, \
            "Wrong value of u_sim function in simulation_with_input.py"    

    if with_plots:
        fig = p.figure()
        p.subplot(2,1,1)
        p.plot(time_sim, x1_sim, time_sim, x2_sim)
        p.subplot(2,1,2)
        p.plot(time_sim, u_sim)
        p.show()

if __name__=="__main__":
    run_demo()

