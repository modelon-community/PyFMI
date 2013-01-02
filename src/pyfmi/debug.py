#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# Copyright (C) 2010 Modelon AB
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

"""
This file contains methods for helping with debugging a simulation.
"""

import numpy as N
import pylab as P
import scipy as S

class DebugInformation:
    pass
    
class CVodeDebugInformation(DebugInformation):
    
    def __init__(self, file_name):
        self.order = []
        self.simulated_time = []
        self.real_time = []
        self.weighted_error = []
        self.events = []
        
        with open(file_name) as file:
            while True:
                row_data = file.readline()
                if row_data.startswith("Time"):
                    row_data = file.readline()
                    while row_data != "\n" and row_data != "":
                        row_data = row_data.split("|")
                        self.simulated_time.append(float(row_data[0].strip()))
                        self.real_time.append(float(row_data[1].strip()))
                        self.order.append(int(row_data[2].strip()))
                        error_data = row_data[3].strip().replace("\n","")
                        error_data = error_data.split(" ")
                        self.weighted_error.append(N.array([abs(float(i)) for i in error_data]))
                        row_data = file.readline()
                elif row_data.startswith("Solver"):
                    self.solver = row_data.split(" ")[-1]
                elif row_data.startswith("State variables:"):
                    self.state_variables = row_data.replace(",","").split(" ")[2:-1]
                elif row_data.startswith("Detected"):
                    self.events.append(float(row_data.split(" ")[-2]))
                if not row_data:
                    break
    
    
    def plot_order(self):
        P.plot(self.simulated_time, self.order,drawstyle="steps-pre")
        P.grid()
        P.xlabel("Time [s]")
        P.ylabel("Order")
        P.title("Order evolution")
        
        self._plot_events()
        
        P.legend(("Order","Events"))
        
        P.show()
        
    def plot_error(self):
        P.semilogy(self.simulated_time, N.array(self.weighted_error))
        P.xlabel("Time [s]")
        P.ylabel("Error")
        P.title("Error evolution")
        P.legend(self.state_variables,loc="lower right")
        P.grid()
        P.show()
        
    def plot_time_distribution(self):
        total_time = N.sum(self.real_time)
        P.plot(self.simulated_time,self.real_time/total_time)
        P.xlabel("Time [s]")
        P.ylabel("Real Time (scaled)")
        
        self._plot_events()
        P.legend(("Time","Events"))
        
        P.grid()
        P.show()
    
    def plot_step_size(self):
        P.semilogy(self.simulated_time,N.diff([0.0]+self.simulated_time),drawstyle="steps-pre")
        P.ylabel("Step-size")
        P.xlabel("Time [s]")
        P.title("Step-size history")
        P.grid()
        P.show()
        
    def _plot_events(self):
        for ev in self.events:
            P.axvline(x=ev,color='r')
