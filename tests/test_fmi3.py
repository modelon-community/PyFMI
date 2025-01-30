from pathlib import Path

this_dir = Path(__file__).parent.absolute()

def test_foo(setup_reference_fmus):
    expected_fmu = Path(this_dir) / 'files' / 'reference_fmus' / '3.0' / 'VanDerPol.fmu'
    assert expected_fmu.exists()
