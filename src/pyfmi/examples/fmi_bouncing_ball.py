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
import pylab as pl
import numpy as np

from pyfmi import load_fmu

curr_dir = os.path.dirname(os.path.abspath(__file__));
path_to_fmus = os.path.join(curr_dir, 'files', 'FMUs','ME1.0')
path_to_fmus2 = os.path.join(curr_dir, 'files', 'FMUs','ME2.0')

def run_demo(with_plots=True, version="2.0"):
    """
    Demonstrates how to use PyFMI / Assimulo for simulation of 
    ME FMUs version 1.0 and 2.0.
    """
    if version == '1.0':
        fmu_name = os.path.join(path_to_fmus,'bouncingBall.fmu')
    else:
        fmu_name = os.path.join(path_to_fmus2,'bouncingBall.fmu')
        
    model = load_fmu(fmu_name)
    
    res = model.simulate(final_time=2.)
    
    # Retrieve the result for the variables
    h_res = res['h']
    v_res = res['v']
    t     = res['time']

    assert np.abs(res.final('h') - (0.0424044)) < 1e-4
    
    # Plot the solution
    if with_plots:
        # Plot the height
        pl.figure()
        pl.clf()
        pl.subplot(2,1,1)
        pl.plot(t, h_res)
        pl.ylabel('Height (m)')
        pl.xlabel('Time (s)')
        # Plot the velocity
        pl.subplot(2,1,2)
        pl.plot(t, v_res)
        pl.ylabel('Velocity (m/s)')
        pl.xlabel('Time (s)')
        pl.suptitle('FMI Bouncing Ball')
        pl.show()

    
if __name__ == "__main__":
    run_demo()
