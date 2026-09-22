#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2026 Modelon AB
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

"""Tests for pyfmi.checks (FMU consistency checks)."""

import numpy as np
import pytest

from pyfmi.checks import check_jacobian, format_jacobian_report, main


@pytest.fixture
def vanderpol():
    from tests.utils import get_fmi3_reference_fmu
    return get_fmi3_reference_fmu("VanDerPol")


@pytest.fixture
def dahlquist():
    from tests.utils import get_fmi3_reference_fmu
    return get_fmi3_reference_fmu("Dahlquist")


class TestCheckJacobian:
    def test_consistent(self, vanderpol):
        res = check_jacobian(vanderpol, times=[1.0, 5.0])
        assert res["ok"] and res["provides_dd"]
        assert [p["time"] for p in res["points"]] == [0.0, 1.0, 5.0]
        for p in res["points"]:
            assert p["rel_norm"] < 1e-6
            assert p["missing"] == 0 and p["wrong"] == 0 and p["undeclared"] == 0
            assert p["dd_nonzero"] == 3      # d(x0)/d(x1), d(x1)/d(x0), d(x1)/d(x1); d(x0)/d(x0) = 0
        assert "OK" in format_jacobian_report(res, "VanDerPol")

    def test_missing_entries_detected(self):
        """Directional derivatives that leave declared entries at zero are reported."""
        from tests.utils import get_fmi3_reference_fmu
        from pyfmi.fmi3 import FMUModelME3

        class ZeroDD(FMUModelME3):
            def get_directional_derivative(self, var_ref, func_ref, v):
                return np.zeros(len(func_ref))

        vanderpol = get_fmi3_reference_fmu("VanDerPol", model_class=ZeroDD)
        res = check_jacobian(vanderpol, times=[1.0])
        assert not res["ok"]
        for p in res["points"]:
            assert p["rel_norm"] > 0.5
            assert p["dd_nonzero"] == 0
            assert p["missing"] == 3
            assert p["missing_rows_by_kind"] == {"x0": 1, "x1": 1}
        assert any(w[2] == 0.0 and abs(w[4]) > 0.1 for w in res["points"][0]["worst"])
        assert "FAILED" in format_jacobian_report(res)

    def test_no_directional_derivatives(self, dahlquist):
        res = check_jacobian(dahlquist)
        assert res["ok"] and not res["provides_dd"] and res["points"] == []
        assert "does not provide" in format_jacobian_report(res)

    def test_state_restored(self, vanderpol):
        vanderpol.initialize(); vanderpol.event_update(); vanderpol.enter_continuous_time_mode()
        x = vanderpol.continuous_states.copy()
        check_jacobian(vanderpol)
        assert vanderpol.time == 0.0
        np.testing.assert_array_equal(vanderpol.continuous_states, x)
        assert vanderpol.force_finite_differences is False

    def test_cli(self, capsys):
        from tests.utils import FMI3_REF_FMU_PATH
        status = main(["jacobian", str(FMI3_REF_FMU_PATH / "VanDerPol.fmu"), "--time", "1.0", "--log-level", "0"])
        assert status == 0
        assert "Jacobian check" in capsys.readouterr().out


class TestCompareResults:
    def test_arrays_and_groups(self):
        from pyfmi.checks import compare_results, format_compare_report
        t = np.linspace(0.0, 10.0, 101)
        names = ["plant.T", "plant.p", "control.PI.limiter.u"]
        Y = np.column_stack([np.sin(t), 1e5 + 10 * t, np.exp(-t)])
        Z = Y.copy()
        Z[:, 0] += 5e-4                 # 5e-4 of max|sin| (0.9996 on this grid)
        Z[:, 1] += 5.0                  # 5 / 1.001e5 = 5e-5
        Z[50:, 2] += 0.5                # 0.5 of max 1 at t = 5
        res = compare_results((t, Y, names), (t, Z, names))
        assert res["n"] == 3 and res["grid"] == (0.0, 10.0, 101)
        assert res["max"] == pytest.approx(0.5)
        g = res["groups"]
        assert set(g) == {"controller", "other"}
        assert g["controller"]["n"] == 1 and g["controller"]["max"] == pytest.approx(0.5)
        assert g["controller"]["worst"][0] == "control.PI.limiter.u" and g["controller"]["worst"][2] == pytest.approx(5.0)
        assert g["other"]["n"] == 2 and g["other"]["max"] == pytest.approx(5e-4, rel=1e-3)
        assert g["other"]["n_above"] == 0
        assert res["variables"]["plant.p"][0] == pytest.approx(5.0 / (1e5 + 100), rel=1e-6)
        assert res["worst"][0][0] == "control.PI.limiter.u"
        report = format_compare_report(res, "a", "b")
        assert "controller" in report and "5.00e-01" in report

        # custom groups, explicit variables, scale floor
        res = compare_results((t, Y, names), (t, Z, names), names=["plant.T"], groups={"thermal": r"\.T$"},
                              scale_floor={"plant.T": 10.0})
        assert list(res["groups"]) == ["thermal"]
        assert res["max"] == pytest.approx(5e-5)

    def test_different_grids_and_events(self):
        """The other result is sampled on the reference grid; at duplicate times (events)
        the last stored value counts."""
        from pyfmi.checks import compare_results
        t_ref = np.linspace(0.0, 1.0, 11)
        y_ref = np.where(t_ref < 0.5, 0.0, 1.0)
        t_oth = np.array([0.0, 0.25, 0.5, 0.5, 0.75, 1.0])
        y_oth = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])      # event at 0.5: pre and post value
        res = compare_results((t_ref, {"s": y_ref}), (t_oth, {"s": y_oth}))
        assert res["max"] == pytest.approx(0.0)
        res = compare_results((t_ref, {"s": y_ref}), (t_oth, {"s": y_oth[::-1]}))
        assert res["max"] == pytest.approx(1.0)

    def test_pyfmi_results_and_cli(self, vanderpol, capsys):
        """Two VanDerPol simulations at different tolerances, as result objects and as files."""
        from pyfmi.checks import compare_results, main
        opts = vanderpol.simulate_options()
        opts["result_file_name"] = "vdp_ref.mat"
        opts["CVode_options"]["rtol"] = 1e-8
        ref = vanderpol.simulate(final_time=5.0, options=opts)
        vanderpol.reset()
        opts["result_file_name"] = "vdp_loose.mat"
        opts["CVode_options"]["rtol"] = 1e-3
        loose = vanderpol.simulate(final_time=5.0, options=opts)

        res = compare_results(ref, loose, names=["x0", "x1"])
        assert res["n"] == 2 and list(res["groups"]) == ["other"]
        assert 1e-5 < res["max"] < 1e-1

        assert main(["compare", "vdp_ref.mat", "vdp_loose.mat", "--vars", "^x", "--threshold", "1.0"]) == 0
        out = capsys.readouterr().out
        assert "Deviation of vdp_loose.mat from vdp_ref.mat" in out and "x" in out
        assert main(["compare", "vdp_ref.mat", "vdp_loose.mat", "--vars", "^x", "--threshold", "1e-9",
                     "--group", "states=^x"]) == 1
        assert "states" in capsys.readouterr().out


class TestOctBlockCheck:
    def test_not_an_oct_fmu(self, vanderpol):
        from pyfmi.checks import enable_oct_block_jacobian_check
        assert enable_oct_block_jacobian_check(vanderpol) is False

    def test_count_warnings(self, tmp_path):
        from pyfmi.checks import count_oct_block_warnings
        log = tmp_path / "x.log"
        log.write_text(
            'FMIL: [WARNING] <SingularJacobian category="warning">Singular Jacobian detected for <value name="dir_block">"36"</value> at t</SingularJacobian>\n'
            'FMIL: [WARNING] <SingularJacobian category="warning">Singular Jacobian detected for <value name="dir_block">"36"</value> at t</SingularJacobian>\n'
            'FMIL: [WARNING] <SingularJacobian category="warning">Singular Jacobian detected for <value name="block">"7"</value></SingularJacobian>\n'
            'FMIL: [WARNING] <SingularJacobian category="warning">Singular Jacobian detected for <value name="dir_block">"9"</value> at t</SingularJacobian>\n')
        assert count_oct_block_warnings(str(log)) == {"36": 2, "9": 1}
