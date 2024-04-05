#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Modelon AB
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

# Testing common functionality

import numpy as np
from pyfmi.common.core import TrajectoryLinearInterpolation, TrajectoryUserFunction


class TestTrajectoryLinearInterpolation:
    def test_shape(self):
        """Test returned shape of TrajectoryLinearInterpolation."""
        t = np.linspace(0, 1, 11)
        x = np.random.rand(11, 3)
        traj = TrajectoryLinearInterpolation(t, x)
        assert traj.eval(0.5).shape == (1, 3)


class TestTrajectoryUserFunction:
    def test_shape_1_dim(self):
        """Testing shape of TrajectoryUserFunction; 1 dim output"""
        traj = TrajectoryUserFunction(lambda x: 5)

        assert traj.eval(1).shape == (1, 1)
        assert traj.eval(np.array([1])).shape == (1, 1)

        assert traj.eval([1, 2]).shape == (2, 1)
        assert traj.eval(np.array([1, 2])).shape == (2, 1)

    def test_shape_multi_dim(self):
        """Testing shape of TrajectoryUserFunction; multi dim output; array"""
        traj = TrajectoryUserFunction(lambda x: np.array([1, 2, 3]))

        assert traj.eval(1).shape == (1, 3)
        assert traj.eval(np.array([1])).shape == (1, 3)

        assert traj.eval([1, 2]).shape == (2, 3)
        assert traj.eval(np.array([1, 2])).shape == (2, 3)

    def test_shape_multi_dim_list(self):
        """Testing shape of TrajectoryUserFunction; multi dim output, list"""
        traj = TrajectoryUserFunction(lambda x: list(range(3)))

        assert traj.eval(1).shape == (1, 3)
        assert traj.eval(np.array([1])).shape == (1, 3)

        assert traj.eval([1, 2]).shape == (2, 3)
        assert traj.eval(np.array([1, 2])).shape == (2, 3)
