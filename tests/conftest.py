import os
import urllib.request
import hashlib
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from pathlib import Path
from typing import Callable
import pytest

files_directory = Path(__file__).parent / 'files'

@pytest.fixture(autouse=True, scope="session")
def test_session_in_tmp_dir_fixture():
    """As our tests produce output in the form of .mat and log files we
    run them in a temporary directory. Long term each test should be isolated
    instead of the session as a whole."""
    with TemporaryDirectory() as tmpdirname:
        workdir = Path(tmpdirname)
        old_cwd = os.getcwd()
        os.chdir(workdir)
        yield Path(workdir)
        os.chdir(old_cwd)

def download_url(url: str, save_file_to: Path):
    """ Download file from URL to 'save_file_to' in chunks. """
    try:
        with urllib.request.urlopen(url) as file_to_download:
            with open(save_file_to, 'wb') as file_handle:
                file_handle.write(file_to_download.read())
    except urllib.request.URLError as e:
        raise Exception(
            "Unable to download reference FMUs, please verify your internet connection is working and" + \
            f" that the URL {url} exists."
            ) from e
    
def download_fmus_from_url_and_cache(
        url: str,
        zip_file_name: str,
        unzip_to: Path,
        file_path_filter: Callable[[str], bool] = None
    ):
    """
    This function downloads FMUs and unpacks them into the test files directory.
    The MD5sum of the URL is checked in order to avoid unnecessary downloading.
    Note that this requires an internet connection to work.

    If 'file_path_filter' is set, only upacks files for which 'file_path_filter(file_path) == True'
    """
    # Simple version of 'cache'
    # Note that we generate the MD5 sum for the URL and not the contents of the zip-file.
    # This is intended and is sufficient for our needs.
    md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    md5_file = Path(unzip_to) / ('metadata_' + md5) # Expected file
    if not md5_file.exists():
        with TemporaryDirectory() as tmpdirname:
            zip_path = Path(tmpdirname) / zip_file_name
            download_url(url, zip_path)

            with ZipFile(zip_path, 'r') as zf:
                for fobj in zf.filelist:
                    if file_path_filter is None or file_path_filter(fobj.filename):
                        zf.extract(fobj, unzip_to)
        with open(md5_file, 'w') as _:
            pass

@pytest.fixture(autouse=True, scope="session")
def setup_reference_fmus():
    """
        This function downloads reference FMUs from the Modelica group and unpacks
        them into the test files directory.
        The MD5sum of the URL is checked in order to avoid unnecessary downloading.
        Note that this requires an internet connection to work.
    """
    zip_file_name = 'reference_fmus.zip'
    unzip_to = files_directory / 'reference_fmus'

    # NOTE: FMI2 Feedthrough.fmu is broken in v0.0.39; use v0.0.38 for FMI2 instead
    # Remove the code below with new reference FMU version
    download_fmus_from_url_and_cache(
        url = "https://github.com/modelica/Reference-FMUs/releases/download/v0.0.39/Reference-FMUs-0.0.39.zip",
        zip_file_name = zip_file_name,
        unzip_to = unzip_to,
        file_path_filter = lambda name: (name.startswith('1.0') or name.startswith('3.0')) and name.endswith('.fmu')
    )
    download_fmus_from_url_and_cache(
        url = "https://github.com/modelica/Reference-FMUs/releases/download/v0.0.38/Reference-FMUs-0.0.38.zip",
        zip_file_name = zip_file_name,
        unzip_to = unzip_to,
        file_path_filter = lambda name: name.startswith('2.0') and name.endswith('.fmu')
    )

@pytest.fixture(autouse=True, scope="session")
def setup_test_fmus():
    download_fmus_from_url_and_cache(
        url = "https://github.com/modelon-community/Test-FMUs/releases/download/v1.0.0/FMUs.zip",
        zip_file_name = 'FMUs.zip',
        unzip_to = files_directory / 'test_fmus'
    )
