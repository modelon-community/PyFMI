# TODO: Remove/rename this file
import sys
from pathlib import Path

this_dir = Path(__file__).parent.absolute()
sys.path.append(str(this_dir))

from utils import setup_reference_fmus

setup_reference_fmus()
expected_fmu = Path(this_dir) / 'files' / 'reference_fmus' / '3.0' / 'VanDerPol.fmu'
if not expected_fmu.exists():
    raise Exception(f"Test setup failed, FMU {expected_fmu} does not exist!")