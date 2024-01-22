#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Modelon AB
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

import os
import numpy as np

from pyfmi import testattr
from pyfmi.tests.test_util import Dummy_FMUModelME2
from scipy.io.matlab.mio import loadmat

assimulo_installed = True
try:
    import assimulo
except ImportError:
    assimulo_installed = False

file_path = os.path.dirname(os.path.abspath(__file__))

if assimulo_installed:
    class Test_FMUModelME2_Estimate:
        
        @testattr(stddist = True)
        def test_quadtank_estimate(self):
            model = Dummy_FMUModelME2([], os.path.join(file_path, "files", "FMUs", "XML", "ME2.0", "QuadTankPack_Sim_QuadTank.fmu"), _connect_dll=False)

            g = model.values[model.get_variable_valueref("qt.g")]
            g1_nmp = model.values[model.get_variable_valueref("qt.g1_nmp")]
            g2_nmp = model.values[model.get_variable_valueref("qt.g2_nmp")]
            k1_nmp = model.values[model.get_variable_valueref("qt.k1_nmp")]
            k2_nmp = model.values[model.get_variable_valueref("qt.k2_nmp")]
            A1 = model.values[model.get_variable_valueref("qt.A1")]
            A2 = model.values[model.get_variable_valueref("qt.A2")]
            A3 = model.values[model.get_variable_valueref("qt.A3")]
            A4 = model.values[model.get_variable_valueref("qt.A4")]
            a3 = model.values[model.get_variable_valueref("qt.a3")]
            a4 = model.values[model.get_variable_valueref("qt.a4")]
            u1_vref = model.get_variable_valueref("u1")
            u2_vref = model.get_variable_valueref("u2")
            a1_vref = model.get_variable_valueref("qt.a1")
            a2_vref = model.get_variable_valueref("qt.a2")

            def f(*args, **kwargs):
                x1 = model.continuous_states[0]
                x2 = model.continuous_states[1]
                x3 = model.continuous_states[2]
                x4 = model.continuous_states[3]

                u1 = model.values[u1_vref]
                u2 = model.values[u2_vref]
                a1 = model.values[a1_vref]
                a2 = model.values[a2_vref]
                    
                sqrt = lambda x: (x)**0.5
                der_x1 = -a1/A1*sqrt(2.*g*x1) + a3/A1*sqrt(2*g*x3) + g1_nmp*k1_nmp/A1*u1
                der_x2 = -a2/A2*sqrt(2.*g*x2) + a4/A2*sqrt(2*g*x4) + g2_nmp*k2_nmp/A2*u2
                der_x3 = -a3/A3*sqrt(2.*g*x3) + (1.-g2_nmp)*k2_nmp/A3*u2
                der_x4 = -a4/A4*sqrt(2.*g*x4) + (1.-g1_nmp)*k1_nmp/A4*u1
                return np.array([der_x1, der_x2, der_x3, der_x4])


            model.get_derivatives = f

            # Load measurement data from file
            data = loadmat(os.path.join(file_path, "files", "Results", "qt_par_est_data.mat"), appendmat=False)

            # Extract data series
            t_meas = data['t'][6000::100,0]-60
            y1_meas = data['y1_f'][6000::100,0]/100
            y2_meas = data['y2_f'][6000::100,0]/100
            y3_meas = data['y3_d'][6000::100,0]/100
            y4_meas = data['y4_d'][6000::100,0]/100
            u1 = data['u1_d'][6000::100,0]
            u2 = data['u2_d'][6000::100,0]

            # Build input trajectory matrix for use in simulation
            u = np.transpose(np.vstack((t_meas,u1,u2)))
            
            # Estimation of 2 parameters
            data = np.vstack((t_meas, y1_meas,y2_meas)).transpose()

            res = model.estimate(parameters=["qt.a1", "qt.a2"],
                                     measurements = (['qt.x1', 'qt.x2'], data), input=(['u1','u2'],u))
            
            
            model.reset()

            # Set optimal values for a1 and a2 into the model
            model.set('qt.a1',res["qt.a1"])
            model.set('qt.a2',res["qt.a2"])

            # Simulate model response with optimal parameters a1 and a2
            res = model.simulate(input=(['u1','u2'],u),start_time=0.,final_time=60)

            assert np.abs(res.final('qt.x1') - 0.07060188) < 1e-3, "Was: " + str(res.final('qt.x1')) + ", expected: 0.07060188"
            assert np.abs(res.final('qt.x2') - 0.06654621) < 1e-3
            assert np.abs(res.final('qt.x3') - 0.02736549) < 1e-3
            assert np.abs(res.final('qt.x4') - 0.02789857) < 1e-3
            assert np.abs(res.final('u1') - 6.0)        < 1e-3
            assert np.abs(res.final('u2') - 5.0)        < 1e-3
