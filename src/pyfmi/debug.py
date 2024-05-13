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

import numpy as np
import pylab as pl

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
                            self.event_indicators.append(np.array([abs(float(i)) for i in event_indicators]))
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
        
        if normalized:
            total_time = np.sum(self.real_time)
            pl.plot(self.simulated_time,self.real_time/total_time)
            pl.ylabel("Real Time (normalized)")
        else:
            pl.plot(self.simulated_time,self.real_time)
            pl.ylabel("Real Time [s]")
        pl.xlabel("Time [s]")
        
        self._plot_events()
        pl.legend(("Time","Events"))
        
        pl.grid()
        pl.show()
        
    def plot_cumulative_time_elapsed(self, log_scale=False):
        cumulative_sum = np.cumsum(self.real_time)
        
        if log_scale:
            pl.semilogy(self.simulated_time, cumulative_sum)
        else:
            pl.plot(self.simulated_time, cumulative_sum)
        pl.xlabel("Time [s]")
        pl.ylabel("Real Time [s]")
        
        self._plot_events()
        pl.legend(("Time","Events"))
        
        pl.grid()
        pl.show()
    
    def plot_step_size(self):
        pl.semilogy(self.simulated_time,np.diff([0.0]+self.simulated_time),drawstyle="steps-pre")
        pl.ylabel("Step-size")
        pl.xlabel("Time [s]")
        pl.title("Step-size history")
        pl.grid()
        pl.show()
        
    def _plot_events(self):
        for ev in self.events:
            pl.axvline(x=ev,color='r')
            
    def plot_event_indicators(self, mask=None, region=None):
        ev_ind = np.array(self.event_indicators)
        time = np.array(self.simulated_time)
        ev_ind_name = np.array(["event_ind_%d"%i for i in range(len(ev_ind[0,:]))])
        
        if region:
            lw = time > region[0]
            up = time < region[1]
            time = time[np.logical_and(lw,up)]
            ev_ind = ev_ind[np.logical_and(lw,up), :]
        
        if mask:
            pl.plot(time, ev_ind[:,mask])
            pl.legend(ev_ind_name[mask])
        else:
            pl.plot(time, ev_ind)
            pl.legend(ev_ind_name)
        pl.grid()
        pl.xlabel("Time [s]")
        pl.title("Event Indicators")
        pl.show()
        
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
                        self.weighted_error.append(np.array([abs(float(i)) for i in error_data]))
                        if len(row_data) > 4:
                            event_indicators = row_data[4].strip().replace("\n","")
                            event_indicators = event_indicators.split(" ")
                            self.event_indicators.append(np.array([abs(float(i)) for i in event_indicators]))
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
        pl.plot(self.simulated_time, self.order,drawstyle="steps-pre")
        pl.grid()
        pl.xlabel("Time [s]")
        pl.ylabel("Order")
        pl.title("Order evolution")
        
        self._plot_events()
        
        pl.legend(("Order","Events"))
        
        pl.show()

    def plot_error(self, threshold=None, region=None, legend=True):
        err = np.array(self.weighted_error)
        time = np.array(self.simulated_time)
        if region:
            lw = time > region[0]
            up = time < region[1]
            time = time[np.logical_and(lw,up)]
            err = err[np.logical_and(lw,up), :]
        if threshold:
            time_points, nbr_vars = err.shape
            mask = np.ones(nbr_vars,dtype=bool)
            for i in range(nbr_vars):
                if np.max(err[:,i]) < threshold:
                    mask[i] = False
            pl.semilogy(time, err[:,mask])
            if legend:
                pl.legend(np.array(self.state_variables)[mask],loc="lower right")
        else:
            pl.semilogy(time, err)
            if legend:
                pl.legend(self.state_variables, loc="lower right")
        pl.xlabel("Time [s]")
        pl.ylabel("Error")
        pl.title("Error evolution")
        pl.grid()
        pl.show()

class Radau5ODEDebugInformation(CVodeDebugInformation):
    pass
