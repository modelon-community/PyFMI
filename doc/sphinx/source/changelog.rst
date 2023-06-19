
==========
Changelog
==========
--- PyFMI-2.10.4 ---
    * Added 'result_handling' = None as option, deprecated 'none'.
    * Fixed so that 'hasattr' works on log nodes.
    * Calls to get continuous states derivatives when there are no such states will no longer result in FMU calls.

--- PyFMI-2.10.3 ---
    * Added method to retrieve the unbounded attribute for real variables in FMI2: get_variable_unbounded.
    * Note: With Assimulo >= 3.4.1 CVode adds supports relative tolerance vectors, see Assimulo README for details.
    * Added partial support for rtol vectors in simulate(): Values need to be equal except for zeros & requires support from used solver.
    * For unbounded states, the simulate method attempts to create a vector of relative tolerances and entries that correspond to unbounded states are set to zero. (FMI2 ME only)

--- PyFMI-2.10.2 ---
    * Corrected version number.

--- PyFMI-2.10.1 ---
    * Changed such that absolute tolerances calculated with state nominals retrieved before initialization will be recalculated with state nominals from after initialization when possible.
    * Added auto correction of retrieved state nominals with illegal values.

--- PyFMI-2.10.0 ---
    * Added shortcut such that get_(real|integer|boolean|string)([]) calls no longer trigger a call to the FMU.
    * Removed the following deprecated functions:
      * pyfmi.fmi.FMUModelCS1, FMUModelME1, FMUModelCS2, FMUModelME2, pyfmi.fmi.FMUModelME1Extended: get_log_file_name: use get_log_filename
      * pyfmi.fmi.FMUModelCS1, FMUModelME1, FMUModelCS2, FMUModelME2, pyfmi.fmi.FMUModelME1Extended: set_fmil_log_level: use set_log_level
    * Removed the following deprecated arguments:
      * pyfmi.fmi.FMUModelCS1, FMUModelME1, FMUModelCS2, FMUModelME2, load_fmu, pyfmi.fmi.FMUModelME1Extended: 'path': use 'fmu' instead
      * pyfmi.fmi.FMUModelCS1, FMUModelME1, FMUModelCS2, FMUModelME2, load_fmu, pyfmi.fmi.FMUModelME1Extended: 'enable_logging': use 'log_level' instead
      * pyfmi.fmi.FMUModelCS1, FMUModelME1Extended.initialize: 'tStart': use 'start_time' instead
      * pyfmi.fmi.FMUModelCS1, FMUModelME1Extended.initialize: 'tStop': use 'stop_time' instead
      * pyfmi.fmi.FMUModelCS1, FMUModelME1Extended.initialize: 'StopTimeDefined': use 'stop_time_defined' instead
      * pyfmi.fmi.FMUModelCS1, FMUModelME1.initialize: 'tolControlled': use 'tolerance_defined' instead
      * pyfmi.fmi.FMUModelCS1, FMUModelME1.initialize: 'relativeTolerance': use 'tolerance' instead
    * Removed the following deprecated attribute:
      * pyfmi.fmi.FMUModelCS1, FMUModelME1, FMUModelCS2, FMUModelME2, load_fmu, pyfmi.fmi.FMUModelME1Extended: 'version': use 'get_version' function instead
    * Note that pyfmi.load_fmu creates instances of pyfmi.fmi.FMUModelCS1, FMUModelME1, FMUModelCS2, FMUModelME2
    * Fixed a crash when using ExplicitEuler with dynamic_diagnostics on models with events.
    * Changed Jacobian to use nominals retrieved via fmu.nominals_continuous_states instead of fmu.get_variable_nominal(<valueref>).
    * Fixed so that malformed log messages do not trigger exceptions and instead the troublesome characters are replaced with a standard replacement character.

--- PyFMI-2.9.8 ---
    * Resolved a bug in the example fmi20_bouncing_ball_native where a step-event is triggered in every timestep.
    * Changed some imports to remove DeprecationWarnings.
    
--- PyFMI-2.9.7 ---
    * Added an argument to ResultDymolaBinary to allow for reading updated
      data from the loaded file.
    * Added option "synchronize_simulation" to allow for synchronizing 
      simulation with (scaled) real-time.

--- PyFMI-2.9.6 ---
    * Added setup.cfg that lists all Python package dependencies in order to run PyFMI.
    * Resolved an issue that would occurr when reading large result files or streams causing the data to be corrupt due to an integer overflow.

--- PyFMI-2.9.5 ---
    * Updated structure of diagnostics data and renamed several of the variables. The top level prefix has also been changed from Diagnostics to @Diagnostics, hence the error check for name clashes has been removed.

--- PyFMI-2.9.4 ---
    * Resolved some tests in need of an update not properly designed for linux.

--- PyFMI-2.9.3 ---
    * Added an internal class variable in ResultHandlerFile to keep track of file position where the data header ends.

--- PyFMI-2.9.2 ---
    * Made error check with 'dynamic_diagnostics' less restrictive to also allow custom result handlers. In 2.9.1 only a binary result handler was allowed.

--- PyFMI-2.9.1 ---
    * Added new simulation option 'dynamic_diagnostics' to better illustrate the purpose of enabling 'logging' when 'result_handling' is set to True.
    * Updated also such that the text file 'model identifier + _debug.txt' is no longer generated when logging to the binary result file.

--- PyFMI-2.9 ---
    * Saving diagnostic data in binary result file instead of log file.
    * Attempts to get continuous states when there are no such states will now return fmi2_status_ok instead of an error.

--- PyFMI-2.8.10 ---
    * Minor updates to exception messages.

--- PyFMI-2.8.9 ---
    * Reverted a fix added in 2.8.8 intended to make sure a log file was not created if no log messages due to potential issues with the implementation.

--- PyFMI-2.8.8 ---
    * Added support for writing result data to streams.
    * Fixed bug with on demand loading with data stored as 32 bit.
    * Fixed segfault when storing data from models with a huge number of
      variables.
    * Loading of FMUs can now be done from an unzipped folder if the argument 'allow_unzipped_fmu' is set to True.
    * The argument 'path' to load_fmu and the different FMI-classes is now deprecated.
    * Added support to log to streams via the keyword argument log_file_name. This is supported for all the FMI-classes as well as the function load_fmu.
    * Improved performance of the Master algorithm.
    * Updated exception types when loading an FMU fails.
    * Delayed creating of a log file. I.e. if there is no log messages
      a log file will not be created.

--- PyFMI-2.8.7 ---
    * Added safety check for updated binary files which can cause
      issues.
    * Fixed so that a matrix of all the result from a binary file can
      be retrieved even if delayed loading is used.

--- PyFMI-2.8.6 ---
    * Fixed so that the written binary file is always consistent (i.e.
      if a simulation aborts, it can still be read)
    * Changed default loading strategiy for binary files, now the
      trajectories are loaded on demand instead of all at the same time.
    * Updated Master algorithm options documentation and fixed result
      file naming.
    * Fixed 'block_initialization' in Master algorithm when using
      Python 3
    * Fixed so that .initial is set properly on ScalarVariable2.
    * Fixed get_variable_nominal when called for valueref of variable.

--- PyFMI-2.8.5 ---
    * Added support for option 'logging' for different ODE solvers.

--- PyFMI-2.8.4 ---
    * Added utility function for determining if the maximum log file
      size has been reached
    * Added support for parsing boolean values in the XML log parser

--- PyFMI-2.8.3 ---
    * Fixed result saving when saving only the "time" variable
    * Exposed the dependencies kind attributes from FMI 2.0

--- PyFMI-2.8.2 ---
    * Added default arguments in the simulation interface (minor)

--- PyFMI-2.8.1 ---
    * Fixed so that the internal event information is saved together
      with the FMU state (when using save / get state).

--- PyFMI-2.8 ---
    * Fixed so that default options are not overriden when setting
      solver options.
    * Improved performance when simulating FMI 2.0 ME FMUs.
    * Building PyFMI now requires that Assimulo is installed.

--- PyFMI-2.7.4 ---
    * Minor fix for save/get state functionality.

--- PyFMI-2.7.3 ---
    * Added support for retrieving relative quantity
    * Fixed pickling of the OptionsBase class
    * Enabled support for serialize/de-serialization of FMU state

--- PyFMI-2.7.2 ---
    * Corrected version number.

--- PyFMI-2.7.1 ---
    * Fixed so that free/terminate methods are called correctly

--- PyFMI-2.7 ---
    * Fixed logging messages being printed to the console during
      instantiation for FMI 1.0
    * Minor encoding issues fixed when retrieving declared types

--- PyFMI-2.6.1 ---
    * Minor fix in handling bytes/str in Python 3

--- PyFMI-2.6 ---
    * Fixed issue with log messages during the FMI methods terminate /
      free instance.
    * Removed caching on the get_variable_nominal method
    * Added a logging module (for parsing XML based FMU logs)
    * Fixed issue with the estimation of directional derivatives when
      the number of outputs was less than the number of states
    * Performance improvements
    * Fixed minor issue when storing the result (https://github.com/modelon-community/PyFMI/issues/21)
    * Added a 'silent' option to the CS simulation options.

--- PyFMI-2.5.7 ---
    * Fixed minor issue in plot GUI for compliance with Python 3.

--- PyFMI-2.5.6 ---
    * Fixed such that instance attributes 'name' and 'raw_name' in class ResultDymolaBinary
      are now attributes that consists of strings instead of bytes in Python 3.
    * Fixed issue with set_string when input was a list of strings in Python 3.
    * Methods _get_types_platform and get_version now returns data of type string
      instead of bytes with Python 3.
    * Fixed other bytes/string incompabilities that caused exceptions with
      Python 3.

--- PyFMI-2.5.5 ---
    * Changed default value of maxh to be computed based on ncp, start
      and stop time according to, maxh=(stop-start)/ncp (ticket:5858)
    * Changed default ncp value from '0' to '500' (ticket:5857)
    * Changed default value for the sparse solver in CVode (if the
      systemsize is >100 and the non-zero pattern is less than 15% then
      a sparse solver is used) (ticket:5666)
    * Changed default value for Jacobian compression (if CVode is used
      and the systemsize is >10 then Jacobian compression is used) (ticket:5666)
    * Added option to specify if the variable descriptions should be
      stored or not in the result file (ticket:5846)
    * Fixed issue with estimating directional derivatives when the
      structure info is not used and the matrix has zero dim (ticket:5836)

--- PyFMI-2.5.4 ---
    * Improved the performance of estimating directional derivatives (ticket:5569)
    * Added support for computing only a subset of interesting columns when considering the cpr seed (ticket:5825)
    * Fixed so that the log file is kept open during the initialization call (ticket:5823)
    * Added support for binary result saving for coupled CS simulations and switched the default storing option to binary (ticket:5820)
    * Changed default value of "linear_correction" to False for coupled CS simulations (ticket:5821)
    * Fixed issue with discrete couplings for coupled CS simulations (ticket:5822)

--- PyFMI-2.5.3 ---
    * Fixed wrong default value for FMUModelME1Extended (ticket:5801)

--- PyFMI-2.5.2 ---
    * Improved relative imports of Assimulo dependent classes (ticket:5798)
    * Fixed unicode symbols in result files (ticket:5797)

--- PyFMI-2.5.1 ---
    * Fixed a number of encode/decoding issues for Python3 (ticket:5786)
    * Forced no copy if the provided array is already correct, minor performance improvement (ticket:5785)
    * Removed a number of C compiler warnings (ticket:5782)
    * Fixed issue with corrupt result files after failed simulations (ticket:5784)
    * Added (hidden) option to only load the XML from an FMU, for testing purposes (ticket:5778)

--- PyFMI-2.5 ---
    * Fixed issue with atol not being updated when rtol is set (ticket:5709)
    * Added check on the nominal values (ticket:5706)
    * Fixed issue with reusing the FD computed Jacobian (ticket:5668)
    * Fixed potential race condition when creating temp directories (ticket:5660)
    * Added a method to retrieve the PyFMI log level (ticket:5639)
    * Made the binary result saving robust to handle incorrect model descriptions (ticket:5624)
    * Fixed issue with using the result filter together with FMI1 (ticket:5623)
    * Improved input handling for FMI2 (ticket:5615)
    * Cleanup of simulation logging (ticket:5614)
    * Fixed simulation logging when there are no states (ticket:5613)
    * Fixed issue with wrong return of time varying variables (ticket:5597)
    * Added functionality to set enumerations with strings (ticket:5587)
    * Changed so that the FMU is only unzipped once (for performance) (ticket:5551)
    * Changed so that the log is stored in memory during load_fmu call (ticket:5550)
    * Added option to limit the maximum size of the log file (ticket:5089)
    * Fixed memory leak when getting the dependency information (ticket:5553)
    * Deprecated get_log_file_name in favour of get_log_filename (ticket:5548)
    * Implemented support for injecting custom logging functionality (ticket:5545)
    * Added the possibility to retrieve unit/display unit name for FMI2 and its value in the display unit (ticket:5537)
    * Added possibility to get a scalar variable directly (ticket:5521)
    * Fixed problem with binary saving (integer start time) (ticket:5496)
    * Updated the interactive info on the load_fmu method (ticket:5495)
    * Changed default file storing method to binary (ticket:5479)
    * Fixed issue with getting parameters when using memory storage option (ticket:5476)
    * Added support for getting the declared type for FMI2 (ticket:5475)
    * Added option to store result files on binary format (ticket:5470)
    * Improved method to retrieve model variables (ticket:5469)
    * Added a prototype of a Master algorithm for coupled ME FMUs (ticket:5438)
    * Fixed so that a "none" result handler can be used for CS (ticket:5403)
    * Removed deprecated FMUModel (ticket:5315)
    * Updated attributes to the initialize methods to be consistent between FMI1 and FMI2. Also added so that setup_experiment is called through FMI2.initialze() if not already called (ticket:5322).
    * Added option "maxh" (maximum step-size) to the Master algorithm (ticket:5396)
    * Fixed bug with step outside simulation region for the Master algorithm (ticket:5397)

--- PyFMI-2.4 ---
    * Fixed a missed encoding of strings, used for Python 3 (ticket:5163)
    * Added timeout option for when simulating CS FMUs (ticket:5313)
    * Added option to specify if the stop time is fixed or not (ticket:5298)
    * Fixed bug where setting the maximum order had no impact (ticket:5212)
    * Added option to use central difference instead of forward differences (ticket:5204)
    * Minor bugfixes and documentation improvements.

--- PyFMI-2.3.1 ---
    * Added caching of model variables when retriving the variables lists (ticket:5007)
    * Added more information about where time is spent in a simulation (ticket:4983)
    * Improved performance when using filters (ticket:4984)

--- PyFMI-2.3 ---
    * Implemented a Master algorithm for simulation of CS FMUS (ticket:4918)
    * Information from the integrator to the log (ticket:4101)
    * Parameter estimation of FMUs (ticket:4461, ticket:4809)
    * Bug fix, plot gui (ticket:4472)
    * Bug fix, pyfmi without assimulo (ticket:4509)
    * Bug fix, handle result (ticket:4658)
    * Bug fix, enum definition (ticket:4740)
    * Bug fix, log name (ticket:4792)
    * Bug fix, enum get/set (ticket:4941)
    * Bug fix, malformed xml (ticket:4888)
    * Allow do steps to be performed in parallel (ticket:4541)
    * Direct acces to low-level FMIL methods (ticket:4542)
    * Performance improvements for get/set (ticket:4566)
    * Fixed output dependencies (ticket:4728, ticket:4762)
    * Fixed derivative dependencies (ticket:4729, ticket:4765)
    * Add option to use finite differences if directional derivatives are not available (ticket:4733)
    * Add support for get/set string (ticket:4798)
    * Added option to disable reloading of simulation results (ticket:4930)

--- PyFMI-2.2 ---
    * Support for sparse representation of matrices (ticket:4306)
    * Update methods for getting variable lists (ticket:4370)
    * Fix for Python 3 (ticket:4386, ticket:4470)
    * Support for get/set FMU state (ticket:4455)
    * Bug fix for result storage (ticket:4460)
    * Bug fix for simulating FMU without states (ticket:4462)
    * Exposed enter/exit initialization mode (ticket:4436)
    * Using PyFMI without Assimulo (ticket:4393)

--- PyFMI-2.1 ---
    * PyFMI Python 3 compliant (ticket:4147)
    * Fix for assert fails in CS simulation (ticket:4244)
    * Methods for retrieving dependency information (ticket:4260)
    * Bug fixes (ticket:4264, ticket:4281)
    * Fix for discard of CS FMUs (ticket:4234)
    * Method for getting real status (ticket:4233)

--- PyFMI-2.0 ---
    * Support for FMI2
    * Added initial to scalar variable (ticket:4146)
    * Support for handling time events directly after intialize (ticket:4122)
    * Fixed saving of enumeration variables (ticket:3778)
    * Added a plot GUI (ticket:1657, ticket:1658, ticket:3703, ticket:4047, ticket:4121)
    * Bug fixes (ticket:3778, ticket:4054, ticket:4053)

--- PyFMI-1.5 ---
    * Added dummy result handler (ticket:3521)
    * Option to implicit euler (ticket:3614)
    * Support for FMI2 RC2 (ticket:3680)

--- PyFMI-1.4.1 ---
    * Improved base result (ticket:3534)

--- PyFMI-1.4 ---
    * Fixed seg fault on Windows (ticket:1947)
    * Added CS example (ticket:2363)
    * Performance improvement when setting inputs (ticket:3032)
    * Changed calling sequence for result handler (ticket:3115)
    * Added option to store result as CSV (ticket:3126)

--- PyFMI-1.3.2 ---
    * Changed the log output from load_fmu (ticket:3030)
    * Fixed enumeration access (ticket:3038)
    * Control of logging (ticket:3013)

--- PyFMI-1.3.1 ---
    * Minor fix in setup script (ticket:2983)

--- PyFMI-1.3 ---
    * Improved result handling (ticket:2864)
    * Changed default values for logging (ticket:2970)
    * Support for LSODAR from Assimulo (ticket:2945)
    * Changed default simulation time (ticket:2910)
    * Added filtering of model variables (ticket:2819)
    * Option to store simulation result in memory (ticket:2813)
    * Added reset method for CS1 (ticket:2724)
    * Fixed get/set negated values (ticket:2758)
    * Improved reset method (ticket:2270)
    * Decode description string to UTF-8 (ticket:2652)
    * Option to store log to file (ticket:2403)
    * Option to get the default experiment data (ticket:2564)
    * Bug fixes (ticket:2489, ticket:2569, ticket:2877, ticket:2916)


--- PyFMI-1.2 ---
    * Added check for empty last error (ticket:2474)
    * Updated bouncingball example (ticket:2478)

--- PyFMI-1.2b1 ---
    * Import and simulation of co-simulation FMUs (ticket:2230)
    * Updated setup script (ticket:2293, ticket:2336)
    * Changed license to LGPL (ticket:2361)
    * Added convenience method getting variable by value ref (ticket:2480)
    * Minor improvements (ticket:2294, ticket:2453)
    * Minor bug fixes (ticket:2314, ticket:2412, ticket:2336)

--- PyFMI-1.1 ---
    * Included FMIL in setup (ticket:1940)
    * Fixed static / shared linking (ticket:2216)

--- PyFMI-1.1b1 ---
    * Changed internals to use FMI Library (FMIL) (ticket:1920)
    * Minor bug fixes (ticket:2203, ticket:1952)
