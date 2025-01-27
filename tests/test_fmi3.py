# TODO: Remove/rename this file
import sys
from pathlib import Path

this_dir = Path(os.path.dirname(__file__)).absolute()
sys.path.append(str(this_dir))

from utils import setup_reference_fmus

setup_reference_fmus()