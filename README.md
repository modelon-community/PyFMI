About
-----------

PyFMI is a package for loading and interacting with Functional Mock-Up Units
(FMUs) both for Model Exchange and Co-Simulation, which are compiled dynamic
models compliant with the Functional Mock-Up Interface ([FMI](https://fmi-standard.org/)).

For a more indebt technical description of the features / functionality see the following [link](https://portal.research.lu.se/portal/files/7201641/pyfmi_tech.pdf).

For citing PyFMI, please use:

<em>Andersson, C, Åkesson, J & Führer, C 2016, PyFMI: A Python Package for Simulation of Coupled Dynamic Models with the Functional Mock-up Interface. Technical Report in Mathematical Sciences, nr. 2, vol. 2016, vol. LUTFNA-5008-2016, Centre for Mathematical Sciences, Lund University</em>.

For information about contributing, see [here](https://github.com/modelon/contributing).

Installation from source
-----------
PyFMI can be built and installed from source, note this requires [FMI library](https://github.com/modelon-community/fmi-library). Additional details are found in the file `setup.py`.

`python setup.py install --fmil-home=/path/to/FMI_Library/`

Here the flag "--fmil-home" points to where FMI Library is installed. The Python package [Assimulo](https://github.com/modelon-community/Assimulo) needs to be installed in order to build from source.

Installation using CONDA
-----------

`conda install -c conda-forge pyfmi`

Note that some examples requires optional dependencies for plotting, they are:
- wxPython
- matplotlib

FMU Import Compliance
-----------
PyFMI is tested daily in several different ways:
* Extensive external testing within the `Optimica Compiler Toolkit` (OCT) provided by [Modelon](https://help.modelon.com/latest/reference/oct/) using all models in Modelon Modelica libraries (35000+ models), exported as FMUs.
* Unit tests from the `tests` directory. These use FMUs generated by
  * OCT
  * JModelica.org
  * FMUs in the example directory are generated via <em>FMU SDK by Qtronic</em>, more information is available in related `README.txt` for these FMUs with proper license text. 
* PyFMI is the default execution engine for [Modelon Impact](https://modelon.com/modelon-impact/).
* Testing with FMUs from [FMI Cross Check](https://github.com/modelica/fmi-cross-check) and commit [55c6704](https://github.com/modelica/fmi-cross-check/commit/55c6704bbcaed3e0f4f788a02af0aba08b7faa4a):
  * 95 FMUs pass, 21 fail. 
  * All the `win64` Dymola FMUs have been tested with PyFMI `2.11.0` and pass with default options.
  * On `Ubuntu 20.04` using `Python 3.9`, `PyFMI 2.11.0`, the following FMUs pass:
     * fmus/2.0/me/linux64/MapleSim/2015.1/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2015.1/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2015.1/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/2015.2/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2015.2/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2015.2/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/2016.1/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2016.1/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2016.1/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/2016.2/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2016.2/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2016.2/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/2018/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2018/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2018/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/2019/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2019/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2019/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/2021.1/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2021.1/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2021.1/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/2021.2/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/MapleSim/2021.2/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/MapleSim/2021.2/Rectifier/Rectifier.fmu
     * fmus/2.0/me/linux64/MapleSim/7.01/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/solidThinking_Activate/2020/ActivateRC/ActivateRC.fmu
     * fmus/2.0/me/linux64/solidThinking_Activate/2020/Arenstorf/Arenstorf.fmu
     * fmus/2.0/me/linux64/solidThinking_Activate/2020/Boocwen/Boocwen.fmu
     * fmus/2.0/me/linux64/solidThinking_Activate/2020/CVloop/CVloop.fmu (simulated with `Radau5ODE` instead of `CVode`)
     * fmus/2.0/me/linux64/solidThinking_Activate/2020/Pendulum/Pendulum.fmu
     * fmus/2.0/me/linux64/SystemModeler/5.0/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/me/linux64/SystemModeler/5.0/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.1/BouncingBall/BouncingBall.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.1/Dahlquist/Dahlquist.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.1/Feedthrough/Feedthrough.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.1/Resource/Resource.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.1/Stair/Stair.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.1/VanDerPol/VanDerPol.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.2/BouncingBall/BouncingBall.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.2/Dahlquist/Dahlquist.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.2/Feedthrough/Feedthrough.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.2/Resource/Resource.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.2/Stair/Stair.fmu
     * fmus/2.0/me/linux64/Test-FMUs/0.0.2/VanDerPol/VanDerPol.fmu
     * fmus/2.0/cs/linux64/20sim/4.6.4.8004/TorsionBar/TorsionBar.fmu
     * fmus/2.0/cs/linux64/EDALab_HIFSuite/2017.05_antlia/b01/b01.fmu
     * fmus/2.0/cs/linux64/EDALab_HIFSuite/2017.05_antlia/des56_original/des56_original.fmu
     * fmus/2.0/cs/linux64/EDALab_HIFSuite/2017.05_antlia/m6502/m6502.fmu
     * fmus/2.0/cs/linux64/EDALab_HIFSuite/2017.05_antlia/uart/uart.fmu
     * fmus/2.0/cs/linux64/MapleSim/2015.1/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2015.1/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2015.1/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/2015.2/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2015.2/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2015.2/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/2016.1/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2016.1/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2016.1/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/2016.2/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2016.2/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2016.2/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/2018/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2018/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2018/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/2019/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2019/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2019/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/2021.1/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2021.1/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2021.1/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/2021.2/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/MapleSim/2021.2/CoupledClutches/CoupledClutches.fmu
     * fmus/2.0/cs/linux64/MapleSim/2021.2/Rectifier/Rectifier.fmu
     * fmus/2.0/cs/linux64/MapleSim/7.01/ControlledTemperature/ControlledTemperature.fmu
     * fmus/2.0/cs/linux64/solidThinking_Activate/2020/ActivateRC/ActivateRC.fmu
     * fmus/2.0/cs/linux64/solidThinking_Activate/2020/Arenstorf/Arenstorf.fmu
     * fmus/2.0/cs/linux64/solidThinking_Activate/2020/Boocwen/Boocwen.fmu
     * fmus/2.0/cs/linux64/solidThinking_Activate/2020/CVloop/CVloop.fmu
     * fmus/2.0/cs/linux64/solidThinking_Activate/2020/DiscreteController/DiscreteController.fmu
     * fmus/2.0/cs/linux64/solidThinking_Activate/2020/Pendulum/Pendulum.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.1/BouncingBall/BouncingBall.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.1/Dahlquist/Dahlquist.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.1/Feedthrough/Feedthrough.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.1/Resource/Resource.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.1/Stair/Stair.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.1/VanDerPol/VanDerPol.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.2/BouncingBall/BouncingBall.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.2/Dahlquist/Dahlquist.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.2/Feedthrough/Feedthrough.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.2/Resource/Resource.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.2/Stair/Stair.fmu
     * fmus/2.0/cs/linux64/Test-FMUs/0.0.2/VanDerPol/VanDerPol.fmu
     * fmus/2.0/cs/linux64/YAKINDU_Statechart_Tools/4.0.4/BouncingBall/BouncingBall.fmu
     * fmus/2.0/cs/linux64/YAKINDU_Statechart_Tools/4.0.4/Feedthrough/Feedthrough.fmu
     * fmus/2.0/cs/linux64/YAKINDU_Statechart_Tools/4.0.4/Stairs/Stairs.fmu 
  * Those that does not pass are believed to fail due to:
    * Linux compliance issues.
    * AMESim FMUs fail since FMUs that need an execution tool are not supported.
* [Reference FMUs](https://github.com/modelica/Reference-FMUs/releases/tag/v0.0.27)
  * All FMUs pass on both `win64` and `linux64` (tested on `Ubuntu 20.04`).
