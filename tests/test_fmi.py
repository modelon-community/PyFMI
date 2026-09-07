#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2026 Modelon AB
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

# Tests for functionality common to several FMI versions

import pytest
import os
from zipfile import ZipFile
import tempfile
import shutil
from pathlib import Path
import dataclasses
from typing import Callable

from pyfmi.fmi import (
    FMUKind,
    FMUException,
    InvalidVersionException,
    load_fmu,
    FMUModelME1,
    FMUModelCS1,
    FMUModelME2,
    FMUModelCS2,
    FMUModelME3,
    FMUModelCS3
)
from pyfmi.test_util import (
    get_examples_folder,
)
from pyfmi.common.core import create_temp_dir

file_path = Path(os.path.dirname(os.path.abspath(__file__)))
FMU_PATHS = file_path / "files" / "FMUs"

FMU_PATHS_ME1_CC = FMU_PATHS / "XML" / "ME1.0" / "CoupledClutches.fmu"
FMU_PATHS_CS1_CC = FMU_PATHS / "XML" / "CS1.0" / "CoupledClutches.fmu"
FMU_PATHS_ME2_CC = FMU_PATHS / "XML" / "ME2.0" / "CoupledClutches.fmu"
FMU_PATHS_CS2_CC = FMU_PATHS / "XML" / "CS2.0" / "CoupledClutches.fmu"

REFERENCE_FMU_PATH = file_path / 'files' / 'reference_fmus'
REFERENCE_FMU_FMI1_PATH = REFERENCE_FMU_PATH / '1.0'
REFERENCE_FMU_FMI2_PATH = REFERENCE_FMU_PATH / '2.0'
REFERENCE_FMU_FMI3_PATH = REFERENCE_FMU_PATH / '3.0'

TEST_FMU_PATH = file_path / 'files' / 'test_fmus'
TEST_FMU_FMI2_ME_PATH = TEST_FMU_PATH / '2.0' / 'me'

PATH_TO_FMU_EXAMPLES = Path(get_examples_folder()) / 'files' / 'FMUs'

@pytest.fixture(params = [
    pytest.param(load_fmu),
    pytest.param(FMUModelME1),
    pytest.param(FMUModelME2),
    pytest.param(FMUModelME3),
    pytest.param(FMUModelCS1),
    pytest.param(FMUModelCS2),
    ]
)
def fmu_loader_for_exception_testing(request):
    return request.param

@dataclasses.dataclass
class FMULoadingTestCase:
    loader: Callable
    path: Path

@pytest.fixture(params = [
    pytest.param(FMULoadingTestCase(FMUModelME1, REFERENCE_FMU_FMI1_PATH / 'me' / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(FMUModelCS1, REFERENCE_FMU_FMI1_PATH / 'cs' / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(FMUModelME2, REFERENCE_FMU_FMI2_PATH / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(FMUModelCS2, REFERENCE_FMU_FMI2_PATH / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(FMUModelME3, REFERENCE_FMU_FMI3_PATH / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(load_fmu, REFERENCE_FMU_FMI1_PATH / 'me' / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(load_fmu, REFERENCE_FMU_FMI1_PATH / 'cs' / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(load_fmu, REFERENCE_FMU_FMI2_PATH / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(load_fmu, REFERENCE_FMU_FMI2_PATH / 'VanDerPol.fmu')),
    pytest.param(FMULoadingTestCase(load_fmu, REFERENCE_FMU_FMI3_PATH / 'VanDerPol.fmu')),
    ]
)
def load_with_path_object(request):
    return request.param

class Test_FMU:
    """Tests that can be parameterized for all FMI versions and/or loading types."""
    def test_invalid_path(self, fmu_loader_for_exception_testing):
        """Test loading an FMU on a path that does not exist."""
        msg = "Could not locate the FMU in the specified directory."
        with pytest.raises(FMUException, match = msg):
            fmu_loader_for_exception_testing("path_that_does_not_exist.fmu")
    
    def test_unzipped_fmu_exception_invalid_dir(self, tmpdir, fmu_loader_for_exception_testing):
        """ Verify that we get an exception if unzipped FMU does not contain modelDescription.xml."""
        err_msg = "Specified fmu path '.*\\' needs to contain a modelDescription.xml according to the FMI specification"
        with pytest.raises(FMUException, match = err_msg):
            fmu_loader_for_exception_testing(str(tmpdir), allow_unzipped_fmu = True)

    def test_unzipped_fmu_exception_is_file(self, fmu_loader_for_exception_testing):
        """ Verify exception is raised if 'fmu' is a file and allow_unzipped_fmu is set to True. """
        err_msg = "Argument named 'fmu' must be a directory if argument 'allow_unzipped_fmu' is set to True."
        fmu_path = tempfile.mktemp(dir = ".")
        with pytest.raises(FMUException, match = err_msg):
            fmu_loader_for_exception_testing(fmu_path, allow_unzipped_fmu = True)

    @pytest.mark.parametrize("fmu_loader, fmu_path",
        [
            (FMUModelME1, PATH_TO_FMU_EXAMPLES / 'ME2.0' / 'bouncingBall.fmu'),
            (FMUModelCS1, PATH_TO_FMU_EXAMPLES / 'CS2.0' / 'bouncingBall.fmu'),
            (FMUModelME2, PATH_TO_FMU_EXAMPLES / 'ME1.0' / 'bouncingBall.fmu'),
            (FMUModelCS2, PATH_TO_FMU_EXAMPLES / 'CS1.0' / 'bouncingBall.fmu'),
            (FMUModelME3, PATH_TO_FMU_EXAMPLES / 'ME1.0' / 'bouncingBall.fmu'),
        ]
    )
    def test_invalid_version(self, fmu_loader, fmu_path):
        """Test using an FMU with the incorrect version."""
        msg = "The FMU could not be loaded."
        with pytest.raises(InvalidVersionException, match = msg):
            fmu_loader(fmu_path, _connect_dll = False)

    def test_load_using_path_object(self, load_with_path_object):
        assert isinstance(load_with_path_object.path, Path)
        load_with_path_object.loader(load_with_path_object.path)

    def test_load_unzipped_using_path_object(self, tmpdir, load_with_path_object):
        shutil.unpack_archive(load_with_path_object.path, format = "zip", extract_dir = tmpdir)
        load_with_path_object.loader(Path(tmpdir), allow_unzipped_fmu = True)

    def test_load_with_log_file_name_as_path(self, load_with_path_object):
        load_with_path_object.loader(
            str(load_with_path_object.path),
            log_file_name = Path("log.txt")
            )

    def test_extract_xml_log_as_path(self, load_with_path_object):
        fmu = load_with_path_object.loader(str(load_with_path_object.path))
        fmu.extract_xml_log(Path("xml_log.xml"))

    def test_result_file_name_as_path(self, load_with_path_object):
        fmu = load_with_path_object.loader(str(load_with_path_object.path))
        opts = fmu.simulate_options()
        opts["ncp"] = 1 # speed
        if opts.get("solver"): # work-around for an FMI1 bug with absolute tolerance adjustments
            opts["solver"] = "ExplicitEuler"
        opts["result_file_name"] = Path("res.mat")
        fmu.simulate(options = opts)

@pytest.mark.parametrize("fmu_loader, fmu_path",
    [
        (FMUModelME1, PATH_TO_FMU_EXAMPLES/ 'ME1.0' / 'bouncingBall.fmu'),
        (FMUModelCS1, PATH_TO_FMU_EXAMPLES/ 'CS1.0' / 'bouncingBall.fmu'),
        (FMUModelME2, PATH_TO_FMU_EXAMPLES/ 'ME2.0' / 'bouncingBall.fmu'),
        (FMUModelCS2, PATH_TO_FMU_EXAMPLES/ 'CS2.0' / 'bouncingBall.fmu'),
        (FMUModelME3, REFERENCE_FMU_FMI3_PATH / "BouncingBall.fmu"),
    ]
)
class TestGetUnpackedFMUPath:
    """Test the get_unpacked_fmu_path function."""
    def test_get_unpacked_fmu_path(self, fmu_loader, fmu_path):
        """Test the default internal unpacking."""
        fmu = fmu_loader(fmu_path)
        unpacked_path = fmu.get_unpacked_fmu_path()
        assert os.path.exists(unpacked_path)
        assert os.path.exists(os.path.join(unpacked_path, "modelDescription.xml"))
        assert os.path.exists(os.path.join(unpacked_path, "binaries"))

    def test_get_unpacked_fmu_path_unpacked_load(self, tmpdir, fmu_loader, fmu_path):
        """Test manual unpacking."""
        shutil.unpack_archive(fmu_path, format = "zip", extract_dir = tmpdir)
        fmu = fmu_loader(str(tmpdir), allow_unzipped_fmu = True)

        unpacked_path = fmu.get_unpacked_fmu_path()
        assert unpacked_path == os.path.abspath(tmpdir)
        assert os.path.exists(os.path.join(unpacked_path, "modelDescription.xml"))
        assert os.path.exists(os.path.join(unpacked_path, "binaries"))


@pytest.mark.parametrize("fmu_loader, fmu_path",
    [
        (load_fmu   , PATH_TO_FMU_EXAMPLES / 'ME1.0' / 'bouncingBall.fmu'),
        (FMUModelME1, PATH_TO_FMU_EXAMPLES / 'ME1.0' / 'bouncingBall.fmu'),
        (load_fmu   , PATH_TO_FMU_EXAMPLES / 'ME2.0' / 'bouncingBall.fmu'),
        (FMUModelME2, PATH_TO_FMU_EXAMPLES / 'ME2.0' / 'bouncingBall.fmu'),
        (load_fmu   , REFERENCE_FMU_FMI3_PATH / 'BouncingBall.fmu'),
        (FMUModelME3, REFERENCE_FMU_FMI3_PATH / 'BouncingBall.fmu'),
        (load_fmu   , PATH_TO_FMU_EXAMPLES / 'CS1.0' / 'bouncingBall.fmu'),
        (FMUModelCS1, PATH_TO_FMU_EXAMPLES / 'CS1.0' / 'bouncingBall.fmu'),
        (load_fmu   , PATH_TO_FMU_EXAMPLES / 'CS2.0' / 'bouncingBall.fmu'),
        (FMUModelCS2, PATH_TO_FMU_EXAMPLES / 'CS2.0' / 'bouncingBall.fmu'),
    ]
)
@pytest.mark.assimulo
class TestUnzippedBouncingBall:
    def _test_unzipped_bouncing_ball(self, fmu_loader, fmu_path, fmu_dir):
        """ Simulates the bouncingBall FMU by unzipping the example FMU before loading."""
        with ZipFile(fmu_path, 'r') as fmu_zip:
            fmu_zip.extractall(path = fmu_dir)

        unzipped_fmu = fmu_loader(fmu_dir, allow_unzipped_fmu = True)
        res = unzipped_fmu.simulate(final_time = 2.0)
        assert res.final("h") == pytest.approx(0.0424044, abs = 1e-2)
    
    def test_create_temp_dir(self, fmu_loader, fmu_path):
        """Test unzipped bouncingBall using FMU unzipped to 'create_temp_dir()'."""
        unzip_dir = create_temp_dir()
        self._test_unzipped_bouncing_ball(fmu_loader, fmu_path, unzip_dir)

    def test_custom_temp_dir(self, tmpdir, fmu_loader, fmu_path):
        """Test unzipped bouncingBall using FMU unzipped to custom temp directory."""
        self._test_unzipped_bouncing_ball(fmu_loader, fmu_path, str(tmpdir))

class Test_LogCategories:
    """Test relating to FMI log categories functionality."""
    @pytest.mark.parametrize("fmu_path", 
        [
            REFERENCE_FMU_FMI2_PATH / "Dahlquist.fmu",
            REFERENCE_FMU_FMI3_PATH / "Dahlquist.fmu",
        ]
    )
    def test_get_log_categories(self, fmu_path):
        """Test getting log categories."""
        fmu = load_fmu(fmu_path)
        log_cats = fmu.get_log_categories()

        assert isinstance(log_cats, dict)
        expected = {
            "logEvents": "Log events",
            "logStatusError": "Log error messages",
        }
        assert log_cats == expected

    def test_get_set_log_categories_fmi2(self):
        """Test setting log categories, fmi2."""
        fmu = load_fmu(REFERENCE_FMU_FMI2_PATH / "Dahlquist.fmu", log_level = 4)
        log_cats = fmu.get_log_categories()

        assert isinstance(log_cats, dict)
        assert log_cats # not empty

        fmu.set_debug_logging(True, log_cats.keys())
        assert fmu.get_log_level() == 3 # changes log level

    def test_get_set_log_categories_fmi3(self):
        """Test setting log categories, fmi3."""
        fmu = load_fmu(REFERENCE_FMU_FMI3_PATH / "Dahlquist.fmu", log_level = 4)
        log_cats = fmu.get_log_categories()

        assert isinstance(log_cats, dict)
        assert log_cats # not empty

        fmu.set_debug_logging(True, log_cats.keys())
        assert fmu.get_log_level() == 4 # does not change log level

    def test_set_debug_logging_off_fmi2(self):
        """Test setting debug logging off, fmi2."""
        fmu = load_fmu(REFERENCE_FMU_FMI2_PATH / "Dahlquist.fmu", log_level = 4)
        fmu.set_debug_logging(False, [])
        assert fmu.get_log_level() == 0 # changes log level

    def test_set_debug_logging_off_fmi3(self):
        """Test setting debug logging off, fmi3."""
        fmu = load_fmu(REFERENCE_FMU_FMI3_PATH / "Dahlquist.fmu", log_level = 4)
        fmu.set_debug_logging(False, [])
        assert fmu.get_log_level() == 4 # does not change log level


class Test_load_fmu_only_XML:
    @pytest.mark.parametrize("fmu_path, test_class",
        [
            (FMU_PATHS_ME1_CC, FMUModelME1),
            (FMU_PATHS_CS1_CC, FMUModelCS1),
            (FMU_PATHS_ME2_CC, FMUModelME2),
            (FMU_PATHS_CS2_CC, FMUModelCS2),
        ]
    )
    def test_load_xml(self, fmu_path, test_class):
        """Test loading only the XML without connecting to the DLL."""
        model = test_class(fmu_path, _connect_dll=False)
        assert model.get_name() == "CoupledClutches"


@pytest.mark.parametrize("fmu_path, expected",
    [
        (REFERENCE_FMU_FMI1_PATH / "me" / "VanDerPol.fmu", "1.0"),
        (REFERENCE_FMU_FMI1_PATH / "cs" / "VanDerPol.fmu", "1.0"),
        (REFERENCE_FMU_FMI2_PATH / "VanDerPol.fmu", "2.0"),
        (REFERENCE_FMU_FMI3_PATH / "VanDerPol.fmu", "3.0"),
    ]
)
def test_get_version(fmu_path, expected):
    """Verify get_version reports the version from a loaded FMU binary."""
    assert load_fmu(fmu_path).get_version() == expected


class Test_get_fmu_kind:
    """Tests for get_fmu_kind, common to all FMI versions."""

    ALL_CLASSES = [
        (FMUModelME1, REFERENCE_FMU_FMI1_PATH / "me" / "VanDerPol.fmu", FMUKind.MODEL_EXCHANGE),
        (FMUModelCS1, REFERENCE_FMU_FMI1_PATH / "cs" / "VanDerPol.fmu", FMUKind.CO_SIMULATION),
        (FMUModelME2, REFERENCE_FMU_FMI2_PATH / "VanDerPol.fmu",        FMUKind.MODEL_EXCHANGE),
        (FMUModelCS2, REFERENCE_FMU_FMI2_PATH / "VanDerPol.fmu",        FMUKind.CO_SIMULATION),
        (FMUModelME3, REFERENCE_FMU_FMI3_PATH / "VanDerPol.fmu",        FMUKind.MODEL_EXCHANGE),
        (FMUModelCS3, REFERENCE_FMU_FMI3_PATH / "VanDerPol.fmu",        FMUKind.CO_SIMULATION),
    ]

    def test_fmu_kind_values(self):
        """Verify FMUKind values, they are relied upon as the load_fmu 'kind' argument."""
        assert FMUKind.MODEL_EXCHANGE == "me"
        assert FMUKind.CO_SIMULATION == "cs"
        assert FMUKind.SCHEDULED_EXECUTION == "se"

    @pytest.mark.parametrize("test_class, fmu_path, expected", ALL_CLASSES)
    def test_get_fmu_kind(self, test_class, fmu_path, expected):
        """Verify each FMU class reports its kind."""
        assert test_class(fmu_path).get_fmu_kind() == expected

    @pytest.mark.parametrize("test_class, fmu_path, expected", ALL_CLASSES)
    def test_get_fmu_kind_without_binary(self, test_class, fmu_path, expected):
        """Verify get_fmu_kind does not require the FMU binary to be loaded."""
        assert test_class(fmu_path, _connect_dll = False).get_fmu_kind() == expected

    @pytest.mark.parametrize("test_class, fmu_path, expected", ALL_CLASSES)
    def test_get_fmu_kind_load_fmu_roundtrip(self, test_class, fmu_path, expected):
        """Verify get_fmu_kind is directly usable as the load_fmu 'kind' argument."""
        kind = test_class(fmu_path).get_fmu_kind()
        assert isinstance(load_fmu(fmu_path, kind = kind), test_class)

@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("fmu_path, kind",
    [
        (REFERENCE_FMU_FMI1_PATH / "me" / "VanDerPol.fmu", "me"),
        (REFERENCE_FMU_FMI1_PATH / "me" / "VanDerPol.fmu", "auto"),
        (REFERENCE_FMU_FMI1_PATH / "cs" / "VanDerPol.fmu", "cs"),
        (REFERENCE_FMU_FMI1_PATH / "cs" / "VanDerPol.fmu", "auto"),

        (REFERENCE_FMU_FMI2_PATH / "VanDerPol.fmu", "me"),
        (REFERENCE_FMU_FMI2_PATH / "VanDerPol.fmu", "cs"),
        (REFERENCE_FMU_FMI2_PATH / "VanDerPol.fmu", "auto"),

        (REFERENCE_FMU_FMI3_PATH / "VanDerPol.fmu", "me"),
        (REFERENCE_FMU_FMI3_PATH / "VanDerPol.fmu", "cs"),
        (REFERENCE_FMU_FMI3_PATH / "VanDerPol.fmu", "auto"),
    ]
)
def test_load_fmu_kind_is_not_case_sensitive(fmu_path, kind, upper):
    kind = kind.upper() if upper else kind
    load_fmu(fmu_path, log_level = 0, kind = kind)