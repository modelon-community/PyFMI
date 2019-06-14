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

"""
This file contains methods for helping with debugging a simulation.
"""

import numpy as N
import scipy as S

class DebugInformation:
    
    def __init__(self, file_name):
        self.simulated_time = []
        self.real_time = []
        self.events = []
        self.event_indicators = []
        self.state_variables = []
        self.file_name = file_name
        
        self._load_data()
        
    def _load_data(self):
        file_name = self.file_name
        
        with open(file_name) as file:
            while True:
                row_data = file.readline()
                if row_data.startswith("Time"):
                    row_data = file.readline()
                    while row_data != "\n" and row_data != "":
                        row_data = row_data.split("|")
                        self.simulated_time.append(float(row_data[0].strip()))
                        self.real_time.append(float(row_data[1].strip()))
                        if len(row_data) > 2:
                            event_indicators = row_data[2].strip().replace("\n","")
                            event_indicators = event_indicators.split(" ")
                            self.event_indicators.append(N.array([abs(float(i)) for i in event_indicators]))
                        row_data = file.readline()
                elif row_data.startswith("Solver"):
                    self.solver = row_data.split(" ")[-1]
                elif row_data.startswith("State variables:"):
                    self.state_variables = row_data.replace(",","").split(" ")[2:-1]
                elif row_data.startswith("Detected"):
                    self.events.append(float(row_data.split(" ")[-2]))
                if not row_data:
                    break
    
    def plot_time_distribution(self, normalized=False):
        import pylab as P
        
        if normalized:
            total_time = N.sum(self.real_time)
            P.plot(self.simulated_time,self.real_time/total_time)
            P.ylabel("Real Time (normalized)")
        else:
            P.plot(self.simulated_time,self.real_time)
            P.ylabel("Real Time [s]")
        P.xlabel("Time [s]")
        
        self._plot_events()
        P.legend(("Time","Events"))
        
        P.grid()
        P.show()
        
    def plot_cumulative_time_elapsed(self, log_scale=False):
        import pylab as P
        
        cumulative_sum = N.cumsum(self.real_time)
        
        if log_scale:
            P.semilogy(self.simulated_time, cumulative_sum)
        else:
            P.plot(self.simulated_time, cumulative_sum)
        P.xlabel("Time [s]")
        P.ylabel("Real Time [s]")
        
        self._plot_events()
        P.legend(("Time","Events"))
        
        P.grid()
        P.show()
    
    def plot_step_size(self):
        import pylab as P
        
        P.semilogy(self.simulated_time,N.diff([0.0]+self.simulated_time),drawstyle="steps-pre")
        P.ylabel("Step-size")
        P.xlabel("Time [s]")
        P.title("Step-size history")
        P.grid()
        P.show()
        
    def _plot_events(self):
        import pylab as P
        
        for ev in self.events:
            P.axvline(x=ev,color='r')
            
    def plot_event_indicators(self, mask=None, region=None):
        import pylab as P
        
        ev_ind = N.array(self.event_indicators)
        time = N.array(self.simulated_time)
        ev_ind_name = N.array(["event_ind_%d"%i for i in range(len(ev_ind[0,:]))])
        
        if region:
            lw = time > region[0]
            up = time < region[1]
            time = time[N.logical_and(lw,up)]
            ev_ind = ev_ind[N.logical_and(lw,up), :]
        
        if mask:
            P.plot(time, ev_ind[:,mask])
            P.legend(ev_ind_name[mask])
        else:
            P.plot(time, ev_ind)
            P.legend(ev_ind_name)
        P.grid()
        P.xlabel("Time [s]")
        P.title("Event Indicators")
        P.show()
        
class ImplicitEulerDebugInformation(DebugInformation):
    pass
    
class ExplicitEulerDebugInformation(DebugInformation):
    pass
        
class CVodeDebugInformation(DebugInformation):
    
    def __init__(self, file_name):
        self.order = []
        self.weighted_error = []
        
        #Call the super class
        DebugInformation.__init__(self, file_name)
        
    def _load_data(self):
        file_name = self.file_name
        
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
                        if len(row_data) > 4:
                            event_indicators = row_data[4].strip().replace("\n","")
                            event_indicators = event_indicators.split(" ")
                            self.event_indicators.append(N.array([abs(float(i)) for i in event_indicators]))
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
        import pylab as P
        
        P.plot(self.simulated_time, self.order,drawstyle="steps-pre")
        P.grid()
        P.xlabel("Time [s]")
        P.ylabel("Order")
        P.title("Order evolution")
        
        self._plot_events()
        
        P.legend(("Order","Events"))
        
        P.show()

    def plot_error(self, threshold=None, region=None, legend=True):
        import pylab as P
        
        err = N.array(self.weighted_error)
        time = N.array(self.simulated_time)
        if region:
            lw = time > region[0]
            up = time < region[1]
            time = time[N.logical_and(lw,up)]
            err = err[N.logical_and(lw,up), :]
        if threshold:
            time_points, nbr_vars = err.shape
            mask = N.ones(nbr_vars,dtype=bool)
            for i in range(nbr_vars):
                if N.max(err[:,i]) < threshold:
                    mask[i] = False
            P.semilogy(time, err[:,mask])
            if legend:
                P.legend(N.array(self.state_variables)[mask],loc="lower right")
        else:
            P.semilogy(time, err)
            if legend:
                P.legend(self.state_variables, loc="lower right")
        P.xlabel("Time [s]")
        P.ylabel("Error")
        P.title("Error evolution")
        P.grid()
        P.show()

class Radau5ODEDebugInformation(CVodeDebugInformation):
    pass
