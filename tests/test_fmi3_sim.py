#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2025 Modelon AB
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

import re
import logging
from io import StringIO
from pathlib import Path

import pytest
import numpy as np

from pyfmi import load_fmu

this_dir = Path(__file__).parent.absolute()
FMI3_REF_FMU_PATH = Path(this_dir) / 'files' / 'reference_fmus' / '3.0'

class TestSimulation:
    """Tests involving simulation of FMUs for FMI 3."""

    def test_simulate(self):
        """Test simulate VDP model and verify the integrity of the results. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        results = fmu.simulate()

        assert results['x0'][0] == 2.0
        assert results['x1'][0] == 0.0
        assert results['x0'][-1] == pytest.approx( 2.00814337)
        assert results['x1'][-1] == pytest.approx(-0.04277047)
        np.testing.assert_equal(results['mu'], np.ones(len(results['x0'])))

    def test_simulate_check_result_members(self):
        """Test simulate VDP model and check accessible data. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        results = fmu.simulate()

        expected_variables = ['time', 'x0', 'der(x0)', 'x1', 'der(x1)', 'mu']

        assert results.keys() == expected_variables

    def test_simulate_change_ncp(self):
        """Test simulate VDP model and change ncp. """
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        opts = fmu.simulate_options()

        opts['ncp'] = 600
        results = fmu.simulate(options = opts)

        expected_variables = ['time', 'x0', 'der(x0)', 'x1', 'der(x1)', 'mu']

        assert results.keys() == expected_variables

        assert all(len(results[v]) == 601 for v in expected_variables)

    def test_simulate_unsupported_result_handler(self):
        """Verify unsupported result handlers raises an exception"""
        fmu = load_fmu(FMI3_REF_FMU_PATH / "VanDerPol.fmu")
        opts = fmu.simulate_options()

        for rh in ['file', 'memory', 'csv']:
            opts['result_handling'] = rh
            msg = f"For FMI3: 'result_handling' set to '{rh}' is not supported. " + \
                   "Consider setting this option to 'binary', 'custom' or None to continue."
            with pytest.raises(NotImplementedError, match = msg):
                results = fmu.simulate(options = opts)

    @pytest.mark.parametrize("ref_fmu, nbr_of_vars",
        [
            (FMI3_REF_FMU_PATH / "Dahlquist.fmu",    4),
            (FMI3_REF_FMU_PATH / "Feedthrough.fmu",  19),
            (FMI3_REF_FMU_PATH / "Resource.fmu",     2),
            (FMI3_REF_FMU_PATH / "Stair.fmu" ,       2),
            (FMI3_REF_FMU_PATH / "VanDerPol.fmu",    6),
        ]
    )
    def test_simulate_reference_fmus(self, ref_fmu, nbr_of_vars):
        """Verify a couple of the reference FMUs simulates without any errors with default settings."""
        fmu = load_fmu(ref_fmu)
        res = fmu.simulate()

        # TODO: Update to verify numerical values later when we have full FMI3 support for Model Exchange.
        # Tests also require updates when we add support for more data types.
        assert len(res.keys()) == nbr_of_vars
