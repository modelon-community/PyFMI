import requests
import hashlib
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from pathlib import Path

files_directory = Path(__file__).parent / 'files'

def setup_reference_fmus():
    """
        This function downloads reference FMUs from the Modelica group and unpacks
        them into the test files directory.
        When FMUs are extract, the MD5 sum is logged to file, in order to identify
        if the setup needs to run again when running repeated tests.

        Note that this requires an internet connection to work.
    """

    zip_file_url = "https://github.com/modelica/Reference-FMUs/releases/download/v0.0.37/Reference-FMUs-0.0.37.zip"
    zip_file_name = 'reference_fmus.zip'
    zip_unzip_to = files_directory / 'reference_fmus'

    def download_url(url, save_path, chunk_size=1024):
        """ Download file from URL to 'save_path'. """
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    def get_zipfile_md5(path_to_zip):
        """ Retrieve the MD5 sum from zipfile specified via 'path_to_zip'. """
        md5 = hashlib.md5()
        with open(path_to_zip, "rb") as f:
            data = f.read() #read file in chunk and call update on each chunk if file is large.
            md5.update(data)
        return md5.hexdigest()

    with TemporaryDirectory() as tmpdirname:
        zip_path = Path(tmpdirname) / zip_file_name
        download_url(zip_file_url, zip_path)

        # Should we change this? We still need to download the ZIP in order to get the MD5sum.
        # Perhaps we can ignore the MD5sum and simply go from the URL and see if the URL (filename etc) has changed
        md5 = get_zipfile_md5(zip_path)

        md5_file = Path(zip_unzip_to) / 'metadata' # Expected file
        use_already_existing = False
        if md5_file.exists():
            with open(md5_file, 'r') as f:
                use_already_existing = md5 == f.read()

        if not use_already_existing:
            with ZipFile(zip_path, 'r') as zf:
                for fobj in zf.filelist:
                    if fobj.filename.startswith('3.0') and fobj.filename.endswith('.fmu'):
                        zf.extract(fobj, zip_unzip_to)
                        with open(md5_file, 'w') as f:
                            f.write(md5)