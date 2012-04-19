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

import pylab as P
import numpy as N

from pyfmi import FMUModel

curr_dir = O.path.dirname(O.path.abspath(__file__));
path_to_fmus = O.path.join(curr_dir, 'files', 'FMUs')

def run_demo(with_plots=True):
    """
    Demonstrates how to use JModelica.org for simulation of FMUs.
    """

    fmu_name = O.path.join(path_to_fmus,'bouncingBall.fmu')
    model = FMUModel(fmu_name)
    
    res = model.simulate(final_time=2.)
    
    #Retrieve the result for the variables
    h_res = res['h']
    v_res = res['v']
    t     = res['time']

    assert N.abs(h_res[-1] - (0.0424044)) < 1e-4, \
            "Wrong value of h_res in fmi_bouncing_ball.py"
    
    #Plot the solution
    if with_plots:
        #Plot the height
        fig = P.figure()
        P.clf()
        P.subplot(2,1,1)
        P.plot(t, h_res)
        P.ylabel('Height (m)')
        P.xlabel('Time (s)')
        #Plot the velocity
        P.subplot(2,1,2)
        P.plot(t, v_res)
        P.ylabel('Velocity (m/s)')
        P.xlabel('Time (s)')
        P.suptitle('FMI Bouncing Ball')
        P.show()

    
if __name__ == "__main__":
    run_demo()
