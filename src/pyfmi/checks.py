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

"""
Consistency checks for Model Exchange FMUs, usable as a library or from the
command line::

    python -m pyfmi.checks jacobian model.fmu [--time T ...] [--threshold 1e-2]

    python -m pyfmi.checks compare ref.mat other.mat [--vars PATTERN] [--group NAME=REGEX ...]

check_jacobian(): compares the FMU's directional derivatives against coloured
finite differences (forward and central) at one or more points on a trajectory
and reports entries that the directional derivatives get wrong or leave out.

compare_results(): deviation of one simulation result from another, per
variable and per variable group (by default controller-like names against the
rest), so that one badly resolved state -- typically an integrator sitting at a
limiter -- does not hide how the plant states agree.
"""

import argparse
import re
import sys

import numpy as np


def _dense(A):
    return A.toarray() if hasattr(A, "toarray") else np.asarray(A, dtype=float)


def _jacobian(model, mode):
    """PyFMI's coloured Jacobian in one mode: 0 = directional derivatives,
    True = forward differences, 2 = central differences."""
    prev = model.force_finite_differences
    model.force_finite_differences = mode
    try:
        return _dense(model._get_A(use_structure_info=True, add_diag=True))
    finally:
        model.force_finite_differences = prev


def _state_kind(name):
    """'a.b.c[3]' -> 'c' (the last name component without indices), used to group rows."""
    return re.sub(r"\[[^\]]*\]", "", name).rsplit(".", 1)[-1]


OCT_BLOCK_CHECK_OPTIONS = {"_block_jacobian_check": True, "_log_level": 4}


def enable_oct_block_jacobian_check(model, tol=1e-4):
    """
    On an OCT-generated FMU (runtime parameters '_block_jacobian_check', '_log_level'
    present) switch on the FMU's own comparison of analytic vs finite-difference block
    Jacobians and verbose logging, before initialization. Returns True if the FMU has
    the options. Afterwards, count_oct_block_warnings(log_file) says which nonlinear
    blocks reported a singular Jacobian in a directional-derivative solve -- the
    typical cause of directional derivatives that are silently zero.
    """
    names = set(model.get_model_variables(causality=0).keys())
    if not set(OCT_BLOCK_CHECK_OPTIONS) <= names:
        return False
    for k, v in OCT_BLOCK_CHECK_OPTIONS.items():
        model.set(k, v)
    if "_block_jacobian_check_tol" in names:
        model.set("_block_jacobian_check_tol", tol)
    return True


def count_oct_block_warnings(log_file):
    """{block id: number of 'SingularJacobian ... dir_block' warnings} from an FMU log file."""
    counts = {}
    with open(log_file, errors="replace") as f:
        for line in f:
            if "SingularJacobian" in line and "dir_block" in line:
                m = re.search(r'name="dir_block">"?([^"<]+)"?', line)
                key = m.group(1) if m else "?"
                counts[key] = counts.get(key, 0) + 1
    return counts


def check_jacobian(model, times=(), threshold=1e-2, rtol=1e-6, solver="CVode", n_worst=10):
    """
    Compares the FMU's directional derivatives (DD) with coloured finite
    differences at the current state and, optionally, at further points along a
    simulated trajectory.

    At each point three Jacobians of the continuous states are evaluated with
    PyFMI's colouring: DD, forward differences (FD) and central differences (CD).
    Entries where FD and CD disagree by more than 10 % are treated as unreliable
    for finite differences (a kink or a discontinuity next to the point) and are
    left out of the verdict. Of the remaining entries the report lists those that
    the FMU's own dependency structure declares but the DD leaves at zero while
    CD is clearly nonzero ("missing"), those that differ by more than 10 %
    ("wrong"), and the relative Frobenius norm ||DD - CD|| / ||CD||. The check
    fails when that norm exceeds 'threshold' at any point.

    Parameters::

        model --
            A loaded FMUModelME2 or FMUModelME3. If it has not been initialized
            the check initializes it (setup_experiment / initialize / event
            iteration / continuous-time mode).

        times --
            Additional points in time at which to compare; the model is
            simulated to each of them with 'solver' at 'rtol' (default: only
            the current state, i.e. t = start time after initialization).

        threshold --
            Relative norm above which a point is reported as failed.
            Default: 1e-2

        n_worst --
            Number of worst entries listed per point.

    Returns::

        A dict with 'ok' (bool), 'provides_dd', 'nx' and 'points', a list of
        one dict per point with the numbers above and the worst entries as
        (row name, column name, dd, fd, cd, declared).
    """
    if not model.get_capability_flags().get("providesDirectionalDerivatives", False):
        return {"ok": True, "provides_dd": False, "nx": model.get_ode_sizes()[0], "points": [],
                "message": "The FMU does not provide directional derivatives; nothing to compare."}

    if model.time is None:
        if hasattr(model, "setup_experiment"):      # FMI 2
            model.setup_experiment(tolerance=rtol)
            model.initialize()
        else:                                        # FMI 3
            model.initialize(tolerance=rtol)
        model.event_update()
        model.enter_continuous_time_mode()

    states = list(model.get_states_list().keys())
    nx = len(states)
    index = {s: i for i, s in enumerate(states)}
    declared = np.zeros((nx, nx), dtype=bool)
    dep, _ = model.get_derivatives_dependencies()
    for i, (_, cols) in enumerate(dep.items()):
        for s in cols:
            if s in index:
                declared[i, index[s]] = True
    np.fill_diagonal(declared, True)   # PyFMI adds the diagonal to the pattern

    result = {"ok": True, "provides_dd": True, "nx": nx, "points": []}
    first = True
    for t_target in [None] + sorted(t for t in times if t is not None):
        if t_target is not None and t_target > model.time:
            opts = model.simulate_options()
            opts["solver"] = solver
            opts["result_handling"] = None
            opts["ncp"] = 0
            opts["initialize"] = False
            opts[solver + "_options"]["rtol"] = rtol
            opts[solver + "_options"]["verbosity"] = 50   # QUIET: no run statistics on stdout
            model.simulate(start_time=model.time, final_time=t_target, options=opts)
        elif t_target is not None and not first:
            continue
        first = False

        t = model.time
        x = model.continuous_states.copy()
        jac = {}
        for key, mode in (("dd", 0), ("fd", True), ("cd", 2)):
            model.time = t
            model.continuous_states = x
            model.get_derivatives()
            jac[key] = _jacobian(model, mode)
        model.time = t
        model.continuous_states = x
        model.get_derivatives()
        A_dd, A_fd, A_cd = jac["dd"], jac["fd"], jac["cd"]

        scale = np.abs(A_cd).max() if A_cd.size else 1.0
        tiny = 1e-8 * max(scale, 1e-300)
        nonzero = (np.abs(A_cd) > tiny) | (np.abs(A_dd) > tiny)
        rel_fd_cd = np.abs(A_fd - A_cd) / (np.abs(A_cd) + tiny)
        unreliable = nonzero & (rel_fd_cd > 0.1)
        usable = nonzero & ~unreliable
        rel_dd_cd = np.abs(A_dd - A_cd) / (np.abs(A_cd) + tiny)
        missing = usable & declared & (np.abs(A_dd) <= tiny) & (np.abs(A_cd) > 1e3 * tiny)
        wrong = usable & (np.abs(A_dd) > tiny) & (rel_dd_cd > 0.1)
        undeclared = (np.abs(A_dd) > 1e3 * tiny) & ~declared
        norm_cd = np.linalg.norm(A_cd[usable]) if usable.any() else 0.0
        rel_norm = (np.linalg.norm((A_dd - A_cd)[usable]) / norm_cd) if norm_cd > 0 else 0.0

        score = np.where(usable, np.abs(A_dd - A_cd) / (np.abs(A_cd) + 1e-3 * scale), 0.0)
        worst = []
        for flat in np.argsort(-score, axis=None)[:n_worst]:
            i, j = np.unravel_index(flat, score.shape)
            if score[i, j] <= 0.0:
                break
            worst.append((states[i], states[j], float(A_dd[i, j]), float(A_fd[i, j]), float(A_cd[i, j]), bool(declared[i, j])))

        rows_missing = {}
        for i in np.where(missing.any(axis=1))[0]:
            k = _state_kind(states[i])
            rows_missing[k] = rows_missing.get(k, 0) + 1

        point = {"time": float(t), "nonzero": int(nonzero.sum()), "dd_nonzero": int((np.abs(A_dd) > tiny).sum()),
                 "cd_nonzero": int((np.abs(A_cd) > tiny).sum()), "declared": int(declared.sum()),
                 "unreliable_fd": int(unreliable.sum()), "missing": int(missing.sum()), "wrong": int(wrong.sum()),
                 "undeclared": int(undeclared.sum()), "rel_norm": float(rel_norm), "ok": bool(rel_norm <= threshold),
                 "missing_rows_by_kind": rows_missing, "worst": worst}
        result["points"].append(point)
        result["ok"] = result["ok"] and point["ok"]
    return result


def format_jacobian_report(res, name=""):
    lines = ["Jacobian check%s: %s" % ((" of " + name) if name else "", "OK" if res["ok"] else "FAILED")]
    if not res.get("provides_dd", True):
        lines.append("  " + res["message"])
        return "\n".join(lines)
    lines.append("  nx = %d, declared dependencies (incl. diagonal) = %d" % (res["nx"], res["points"][0]["declared"] if res["points"] else 0))
    for p in res["points"]:
        lines.append("  t = %-10g %s  ||DD-CD||/||CD|| = %.2e   nonzero: DD %d, CD %d;  missing in DD %d, wrong %d, "
                     "undeclared in DD %d, FD-unreliable (kinks) %d" % (
                         p["time"], "ok    " if p["ok"] else "FAILED", p["rel_norm"], p["dd_nonzero"], p["cd_nonzero"],
                         p["missing"], p["wrong"], p["undeclared"], p["unreliable_fd"]))
        if p["missing_rows_by_kind"]:
            lines.append("      rows with missing DD entries, by state kind: " +
                         ", ".join("%s: %d" % kv for kv in sorted(p["missing_rows_by_kind"].items(), key=lambda kv: -kv[1])))
        for (r, c, dd, fd, cd, decl) in p["worst"]:
            lines.append("      d(%s)/d(%s): DD %.6g  FD %.6g  CD %.6g%s" % (r, c, dd, fd, cd, "" if decl else "  (not declared)"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pyfmi.checks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jacobian", help="directional derivatives vs finite differences")
    j.add_argument("fmu")
    j.add_argument("--time", type=float, nargs="*", default=[], help="additional points in time to compare at (simulated to with CVode)")
    j.add_argument("--threshold", type=float, default=1e-2, help="relative norm above which the check fails (default 1e-2)")
    j.add_argument("--rtol", type=float, default=1e-6)
    j.add_argument("--worst", type=int, default=10, help="number of worst entries to list per point")
    j.add_argument("--log-level", type=int, default=2)
    j.add_argument("--oct-block-check", action="store_true",
                   help="OCT FMUs: also switch on the FMU's internal block-Jacobian check and report nonlinear "
                        "blocks whose directional-derivative solve had a singular Jacobian (log in <fmu>.jaccheck.log)")
    c = sub.add_parser("compare", help="deviation of one result file from another, per variable group")
    c.add_argument("ref", help="reference result (.mat or .txt)")
    c.add_argument("other")
    c.add_argument("--vars", default=None, help="regex selecting the variables to compare (default: all common ones)")
    c.add_argument("--group", action="append", default=[], metavar="NAME=REGEX",
                   help="variable group by regex (repeatable; default: controller-like names vs. the rest)")
    c.add_argument("--threshold", type=float, default=1e-3)
    c.add_argument("--floor", type=float, default=1e-6, help="lower bound of the per-variable scale")
    c.add_argument("--worst", type=int, default=10)
    args = ap.parse_args(argv)

    if args.cmd == "jacobian":
        import os
        from pyfmi import load_fmu
        log_file = os.path.basename(args.fmu) + ".jaccheck.log"
        if args.oct_block_check:
            model = load_fmu(args.fmu, log_level=4, log_file_name=log_file)
            if not enable_oct_block_jacobian_check(model):
                print("--oct-block-check: not an OCT FMU (no '_block_jacobian_check' parameter); ignored")
                args.oct_block_check = False
        else:
            model = load_fmu(args.fmu, log_level=args.log_level)
        res = check_jacobian(model, times=args.time, threshold=args.threshold, rtol=args.rtol, n_worst=args.worst)
        print(format_jacobian_report(res, args.fmu))
        if args.oct_block_check:
            counts = count_oct_block_warnings(log_file)
            if counts:
                print("  OCT block check: singular Jacobian in directional-derivative solves of block(s) " +
                      ", ".join("%s (%d times)" % kv for kv in sorted(counts.items())) + "  [%s]" % log_file)
            else:
                print("  OCT block check: no singular directional-derivative block solves reported  [%s]" % log_file)
        return 0 if res["ok"] else 1

    ref, other = _FileResult(args.ref), _FileResult(args.other)
    names = [n for n in ref.keys() if n in set(other.keys())]
    if args.vars:
        names = [n for n in names if re.search(args.vars, n)]
    groups = dict(g.split("=", 1) for g in args.group) if args.group else None
    res = compare_results(ref, other, names=names, groups=groups, scale_floor=args.floor,
                          threshold=args.threshold, n_worst=args.worst)
    print(format_compare_report(res, args.ref, args.other))
    return 0 if res["max"] <= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------- trajectories

DEFAULT_GROUPS = {
    # a PI/PID integrator whose output sits at a limiter feeds nothing back to the plant,
    # so its local error accumulates unchecked; such states dominate max-norm deviations
    "controller": r"(?i)control|regulat|\bPI\b|PID|limiter|integrator|antiwindup|anti_windup|\.I\.|\.x_?i\b",
}


def _as_arrays(result, names=None):
    """(t, {name: values}) from a PyFMI result object, a (t, {name: values}) pair or a
    (t, Y, names) triple."""
    if isinstance(result, tuple) and len(result) == 3:
        t, Y, nms = result
        return np.asarray(t, float), {n: np.asarray(Y)[:, k] for k, n in enumerate(nms)}
    if isinstance(result, tuple) and len(result) == 2:
        t, d = result
        return np.asarray(t, float), {n: np.asarray(v, float) for n, v in d.items()}
    t = np.asarray(result["time"], float)
    if names is None:
        names = [n for n in result.keys() if n != "time"] if hasattr(result, "keys") else []
    return t, {n: np.asarray(result[n], float) for n in names}


def compare_results(ref, other, names=None, groups=None, scale_floor=None, threshold=1e-3, n_worst=10,
                    grid=None):
    """
    Deviation of 'other' from 'ref', per variable and per variable group.

    Both results are sampled on a common grid (by default the reference's time
    points inside the common interval; the last stored value is taken at
    duplicate times, i.e. the post-event value) and compared as
    |other - ref| / scale, with scale = max(max|ref|, floor) per variable.

    Parameters::

        ref, other --
            PyFMI result objects (anything indexable by variable name with a
            "time" entry), (time, {name: values}) pairs or (time, Y, names).

        names --
            Variables to compare (default: all variables present in both).

        groups --
            {group name: regex} deciding the group of a variable by the first
            regex that matches its name; unmatched variables go to "other".
            Default: DEFAULT_GROUPS (controller-like names vs. the rest).

        scale_floor --
            {name: floor} or a scalar: lower bound of the per-variable scale
            (e.g. the state nominals, or the absolute tolerance). Default: 1e-6.

        threshold --
            Deviation above which a variable is counted in 'n_above'.

        grid --
            Explicit time grid to compare on (default: the reference's times).

    Returns::

        dict with 'max' (overall), 'groups' {name: {'max', 'median', 'n', 'n_above',
        'worst': (variable, deviation, time)}}, 'variables' {name: (deviation, time)},
        'worst' [(variable, deviation, time, group), ...] and 'grid' (t0, tf, n).
    """
    t_a, A = _as_arrays(ref, names)
    t_b, B = _as_arrays(other, names)
    common = [n for n in A if n in B] if names is None else list(names)
    if groups is None:
        groups = DEFAULT_GROUPS
    if grid is None:
        t0, tf = max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])
        grid = t_a[(t_a >= t0) & (t_a <= tf)]
    grid = np.asarray(grid, float)

    def sample(t, y):
        # last stored value at each grid time (post-event value where an event lands on it)
        idx = np.clip(np.searchsorted(t, grid * (1 + 1e-12) + 1e-300, side="right") - 1, 0, len(t) - 1)
        return y[idx]

    def floor_of(n):
        if scale_floor is None:
            return 1e-6
        if isinstance(scale_floor, dict):
            return float(scale_floor.get(n, 1e-6))
        return float(scale_floor)

    def group_of(n):
        for g, pattern in groups.items():
            if re.search(pattern, n):
                return g
        return "other"

    variables, per_group = {}, {}
    for n in common:
        ya, yb = sample(t_a, A[n]), sample(t_b, B[n])
        scale = max(float(np.max(np.abs(ya))) if ya.size else 0.0, floor_of(n), 1e-300)
        e = np.abs(yb - ya) / scale
        k = int(np.argmax(e)) if e.size else 0
        dev, t_dev = (float(e[k]), float(grid[k])) if e.size else (0.0, float("nan"))
        variables[n] = (dev, t_dev)
        per_group.setdefault(group_of(n), []).append((n, dev, t_dev))

    out_groups = {}
    for g, items in per_group.items():
        devs = np.array([d for _, d, _ in items])
        worst = max(items, key=lambda it: it[1])
        out_groups[g] = {"n": len(items), "max": float(devs.max()), "median": float(np.median(devs)),
                         "n_above": int((devs > threshold).sum()), "worst": worst}
    worst = sorted(((n, d, t, group_of(n)) for n, (d, t) in variables.items()), key=lambda it: -it[1])[:n_worst]
    return {"max": max((d for d, _ in variables.values()), default=0.0), "groups": out_groups,
            "variables": variables, "worst": worst, "threshold": threshold,
            "grid": (float(grid[0]) if grid.size else float("nan"), float(grid[-1]) if grid.size else float("nan"), int(grid.size)),
            "n": len(common)}


def format_compare_report(res, ref_name="reference", other_name="other"):
    lines = ["Deviation of %s from %s: max %.2e over %d variables, %d grid points in [%g, %g]" % (
        other_name, ref_name, res["max"], res["n"], res["grid"][2], res["grid"][0], res["grid"][1])]
    for g, r in sorted(res["groups"].items(), key=lambda kv: -kv[1]["max"]):
        lines.append("  %-12s n = %4d  max %.2e  median %.2e  > %.0e: %d   worst: %s @ t = %g" % (
            g, r["n"], r["max"], r["median"], res["threshold"], r["n_above"], r["worst"][0], r["worst"][2]))
    for n, d, t, g in res["worst"]:
        lines.append("      %.2e  %-12s %s @ t = %g" % (d, g, n, t))
    return "\n".join(lines)


def _load_result_file(path):
    from pyfmi.common.io import ResultDymolaBinary, ResultDymolaTextual
    return ResultDymolaBinary(path) if path.endswith(".mat") else ResultDymolaTextual(path)


class _FileResult:
    """Indexable view of a result file for compare_results()."""
    def __init__(self, path):
        self._r = _load_result_file(path)

    def keys(self):
        return [n for n in self._r.get_variable_names()]

    def __getitem__(self, name):
        return self._r.get_variable_data(name).x if name != "time" else self._r.get_variable_data("time").x
