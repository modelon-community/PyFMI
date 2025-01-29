import urllib.request
import hashlib
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from pathlib import Path
import pytest

files_directory = Path(__file__).parent / 'files'

@pytest.fixture(autouse=True, scope="session")
def setup_reference_fmus():
    """
        This function downloads reference FMUs from the Modelica group and unpacks
        them into the test files directory.
        The MD5sum of the URL is checked in order to avoid unnecessary downloading.
        Note that this requires an internet connection to work.
    """

    def download_url(url, save_file_to, chunk_size=1024):
        """ Download file from URL to 'save_file_to' in chunks. """
        with urllib.request.urlopen(url) as file_to_download:
            with open(save_file_to, 'wb') as file_handle:
                file_handle.write(file_to_download.read())

    zip_file_url = "https://github.com/modelica/Reference-FMUs/releases/download/v0.0.37/Reference-FMUs-0.0.37.zip"
    zip_file_name = 'reference_fmus.zip'
    zip_unzip_to = files_directory / 'reference_fmus'

    # Simple version of 'cache'
    # Note that we generate the MD5 sum for the URL and not the contents of the zip-file.
    # This is intended and is sufficient for our needs.
    md5 = hashlib.md5(zip_file_url.encode("utf-8")).hexdigest()
    md5_file = Path(zip_unzip_to) / 'metadata' # Expected file
    use_already_existing = False
    if md5_file.exists():
        with open(md5_file, 'r') as f:
            use_already_existing = md5 == f.read()

    if not use_already_existing:
        with TemporaryDirectory() as tmpdirname:
            zip_path = Path(tmpdirname) / zip_file_name
            download_url(zip_file_url, zip_path)

            with ZipFile(zip_path, 'r') as zf:
                for fobj in zf.filelist:
                    # For now, only unpack FMI 3.0 FMUs
                    if fobj.filename.startswith('3.0') and fobj.filename.endswith('.fmu'):
                        zf.extract(fobj, zip_unzip_to)
                        with open(md5_file, 'w') as f:
                            f.write(md5)