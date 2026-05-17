project(
  'PyFMI',
  ['c', 'cython'],
  version: '3.0.0.dev0',
  meson_version: '>=1.4'
)

py = import('python').find_installation(pure: false)
fs = import('fs')
cc = meson.get_compiler('c')

numpy_inc = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

# Parent of the assimulo package so Cython resolves `cimport assimulo.problem`.
assimulo_pkg_parent = run_command(
  py, ['-c', 'import assimulo, os; print(os.path.dirname(os.path.dirname(assimulo.__file__)))'],
  check: true
).stdout().strip()

# numpy_inc is passed via -I rather than include_directories() because under
# pip build isolation it lives outside the source tree, which meson rejects
# for include_directories().
if cc.get_argument_syntax() == 'msvc'
  common_c_args = [
    '-I' + numpy_inc,
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '/O2',
  ]
else
  common_c_args = [
    '-I' + numpy_inc,
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-fno-strict-aliasing',
    '-O2',
  ]
endif

fmil_prefix = get_option('fmil_prefix')
if fmil_prefix == ''
  fmil_prefix = run_command(
    py, ['-c', 'import os, sys; sys.stdout.write(os.environ.get("FMIL_HOME", ""))'],
    check: true
  ).stdout().strip()
endif
if fmil_prefix == ''
  error('FMIL not found. Set -Dfmil_prefix=/path/to/fmil or env FMIL_HOME.')
endif

fmil_incdir = join_paths(fmil_prefix, 'include')
if not fs.is_dir(fmil_incdir)
  error('FMIL include directory not found at @0@'.format(fmil_incdir))
endif

fmil_libdirs = []
foreach d : ['lib', 'lib64', 'bin']
  p = join_paths(fmil_prefix, d)
  if fs.is_dir(p)
    fmil_libdirs += p
  endif
endforeach
if fmil_libdirs.length() == 0
  error('No FMIL lib directory found under @0@'.format(fmil_prefix))
endif

fmil_name = get_option('fmil_name')
fmil_lib  = cc.find_library(fmil_name, dirs: fmil_libdirs, required: true)

cython_include_dirs = [
  '-I', meson.current_source_dir(),
  '-I', join_paths(meson.current_source_dir(), 'src'),
  '-I', join_paths(meson.current_source_dir(), 'src/pyfmi'),
  '-I', assimulo_pkg_parent,
]
cython_common_args = ['-3', '-X', 'language_level=3str'] + cython_include_dirs

# --- RPATH for non-wheel Linux installs: point at the FMIL libdir so the
# shared library is found at runtime. Wheel builds get $ORIGIN added in C4
# once the FMIL .so is bundled inside the package. ---
ext_install_rpath = ''
if host_machine.system() != 'windows'
  ext_install_rpath = fmil_libdirs[0]
endif

with_openmp     = get_option('openmp')
openmp_c_args   = with_openmp ? ['-fopenmp'] : []
openmp_link_args = with_openmp ? ['-fopenmp'] : []

ext_inc = [
  include_directories('src', 'src/pyfmi'),
  include_directories(fmil_incdir),
]

plain_modules = [
  { 'name': 'fmi_base',     'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi_base.pyx' },
  { 'name': 'fmi',          'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi.pyx' },
  { 'name': 'fmi1',         'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi1.pyx' },
  { 'name': 'fmi2',         'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi2.pyx' },
  { 'name': 'fmi3',         'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi3.pyx' },
  { 'name': 'fmi_util',     'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi_util.pyx' },
  { 'name': 'fmi_extended', 'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi_extended.pyx' },
  { 'name': 'fmi_coupled',  'subdir': 'pyfmi',            'src': 'src/pyfmi/fmi_coupled.pyx' },
  { 'name': 'util',         'subdir': 'pyfmi',            'src': 'src/pyfmi/util.pyx' },
  { 'name': 'test_util',    'subdir': 'pyfmi',            'src': 'src/pyfmi/test_util.pyx' },
  { 'name': 'assimulo_interface',      'subdir': 'pyfmi/simulation', 'src': 'src/pyfmi/simulation/assimulo_interface.pyx' },
  { 'name': 'assimulo_interface_fmi1', 'subdir': 'pyfmi/simulation', 'src': 'src/pyfmi/simulation/assimulo_interface_fmi1.pyx' },
  { 'name': 'assimulo_interface_fmi2', 'subdir': 'pyfmi/simulation', 'src': 'src/pyfmi/simulation/assimulo_interface_fmi2.pyx' },
  { 'name': 'assimulo_interface_fmi3', 'subdir': 'pyfmi/simulation', 'src': 'src/pyfmi/simulation/assimulo_interface_fmi3.pyx' },
]

foreach m : plain_modules
  py.extension_module(
    m['name'],
    sources: [m['src']],
    subdir: m['subdir'],
    include_directories: ext_inc,
    c_args: common_c_args,
    cython_args: cython_common_args,
    dependencies: [fmil_lib],
    install_rpath: ext_install_rpath,
    build_rpath:   ext_install_rpath,
    install: true,
  )
endforeach

# master.pyx branches on WITH_OPENMP at Cython compile time; the C compiler
# also needs -fopenmp when openmp is enabled (added below via openmp_c_args).
master_cython_args = cython_common_args + [
  '-E', 'WITH_OPENMP=' + (with_openmp ? 'True' : 'False'),
]

py.extension_module(
  'master',
  sources: ['src/pyfmi/master.pyx'],
  subdir: 'pyfmi',
  include_directories: ext_inc,
  c_args: common_c_args + openmp_c_args,
  cython_args: master_cython_args,
  dependencies: [fmil_lib],
  link_args: openmp_link_args,
  install_rpath: ext_install_rpath,
  build_rpath:   ext_install_rpath,
  install: true,
)

install_subdir('src/pyfmi', install_dir: py.get_install_dir())
