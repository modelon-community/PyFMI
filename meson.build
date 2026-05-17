project(
  'PyFMI',
  ['c', 'cython'],
  version: '3.0.0.dev0',
  meson_version: '>=1.4'
)

py = import('python').find_installation(pure: false)

# C1 skeleton: install the pure-Python package tree only. Cython extension
# modules are still produced by the legacy setup.py during the C1 -> C3
# transition. A pip install via mesonpy at this commit will produce a wheel
# without compiled extensions — that is intentional for the checkpoint.

install_subdir('src/pyfmi', install_dir: py.get_install_dir())
