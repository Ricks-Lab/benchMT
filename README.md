# benchMT  -  SETI multi-threaded MB/AP Benchmark Tool

 Download the latest release [v1.6.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v1.6.0)

 This tool will extract the total number of CPU cores/threads and GPU platforms from the user's
 environment and utilize them in running a list of apps/args specified in the benchCFG file.
 Using less than the total number of CPU threads can be specified in the command line.  This
 tool will read a list of MB/AP apps/args from the BenchCFG file and search for the specified
 MB/AP apps in the APP_CPU and APP_GPU directories to validate and determine platform.  It will
 then leverage allocated threads, as specified, to run all benchmark jobs, storing results in
 the testData directory.  Use the *--help* option to get a description of valid command line
 arguments. In support of automation, some command line arguments can be specified as modes in
 the BenchCFG file.

 By default, a summary list of all jobs will update in the display as the program progresses.  If
 there are a large number of jobs, then this display may not be useful and the *--display_slots*
 option can be used to display the status of each slot as the program progresses.  In some cases,
 there will be too many slots to display, and the *--display_compact* option can used to further
 optimize the progress display.

 You may need to use the *--boinc_home* command option to specify the BOINC home directory, which
 is required, since boinccmd is used. An alternative BenchCFG file can be specified with the
 command line option *--cfg_file filename*.

 The *--lsgpu* command option can be used to display information and capability of all installed
 GPUs. The *--purge_kernels* can be used to purge all compiled kernels from the *benchMT* working
 directory.

 All WUs in the WU_test directory will be used in the creation of jobs to be run, unless the
 *--std_signals* option is used, in which case, WUs in the WU_std_signal will be used.  The
 APPS_GPU and APPS_CPU directories can have more apps than are specified to run in the BenchCFG
 file, but must contain apps specified in BenchCFG.  The APPS_REF directory must contain a single
 CPU reference app with a file prefix of "ref-cpu.".  The stock CPU app is suggested, as this is
 only used to test integrity of the results.  Elapsed time analysis is expected to be limited to
 apps/arg combinations specified in BenchCFG.  The generation of reference results can be skipped
 with the *--no_ref* option or forced with the *--force_ref* option. The *--energy* option can be
 used if your system has amdgpu drivers with compatible GPUs to give the energy used in running a
 task.  In order to correctly associate a GPU card number with a BOINC device number, you must
 specify this with the *--devmap B:C,B2:C2* option.  I know of no robust way to make this mapping
 other than manually running each card individually and observing which card is being used.  If
 you are running an AstroPulse app, you must specify the *--astropulse* option in order for it to
 run properly.

 The results will be stored in a unique subdir of the testData directory. There is an overall run
 log txt file, a psv file useful for importing into an analytics tools, and the sah and stderr
 files for each job run. A run name can be specified with the *--run_name* command line option.
 This name will be included in the name of the testData subdirectory for the current run.
    
## New in this Release  -  [v2.0.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v2.0.0)
* Complete rewrite of the core to integrate GpuItem object that was previously only used in energy mode.
* Implemented a more robust check of compute and energy compatibility for all installed GPUs. Switched
from lshw to lspci and clinfo. Not working for [Intel GPUs](https://github.com/Ricks-Lab/benchMT/issues/6).
* Many core improvements for better robustness and style.  Added reST docstrings for better supportability
and improved type checking by PyCharm.
* Extend energy mode to cover Nvidia, in addition to AMD GPU's.  Added max-power to psv and txt reports.
* *--noBS* on the command line will allow benchMT to function without a BOINC installation.
* Check devmap and give more details to help user understand mapping of BOINC Devices to Driver Card Numbers.
* Added *--purge_kernels* command line option to purge cached kernel files from working directory.
* Added *--lsgpu* command line option to display information on all installed GPUs.
* Reading BOINC's coproc_info.xml as a possible way to automate device to card mapping.  Not implemented.

## Development Plans and Known Limitations
* Doesn't work in console of server install of Ubuntu.  Need to investigate this.
* Currently, running more than one job at a time on a single GPU is not supported. 
* The clinfo output for [Intel GPUs](https://github.com/Ricks-Lab/benchMT/issues/6) doesn't seem to have
a pcie id indicator based on the data I have on 1 card.
* Need to find a way to [automate mapping](https://github.com/Ricks-Lab/benchMT/issues/8) of BOINC device
number to Linux card number or PCIE ID.  Results from reading coproc_info.html are available in *--debug*
mode for experimentation.
* This package doesn't include a utility to compare AstroPulse results to a reference. If you know where
a Linux version of this exists, please let me know.
* Special NV app outputs results to stdout which corrupts the benchMT updating table.
* When no GPU work is selected, benchMT still validates GPU related command line arguments.

## History
#### New in Previous Release  -  [v1.6.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v1.6.0)
* Complete rewrite of commandline/config file option parsing.  Original got complex and buggy.
* Support execution and time/energy metrics for AstroPulse apps/wus.  Still no working results comparison utility,
so comparison to reference results not possible.

#### New in Previous Release  -  [v1.5.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v1.5.0)
* Implemented checks of Python and Kernel version to verify compatibility.
* Implemented more robust system calls.
* Fixed an error in parsing lshw output to extract GPU names and added error checking.
* When specifying specific GPUs to be used with *--gpu_devices*, max_gpus is now automatically set.
* Implemented Energy metric for GPUs using amdgpu drivers.
* Code robustness improvements (path joins and key checks).
* Run completion message now includes the location of the results.

#### New in Previous Release  -  [v1.4.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v1.4.0)
* Write run_name to the psv file.  Useful when wanting to analyze the data from multiple runs.
* Include Nvidia stock MB app in the distribution.
* Include error message when job fails to spawn.

#### New in Previous Release  -  [v1.3.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v1.3.0)
* SETI MB apps are now run without a shell, using shlex to parse args for the subprocess command.
* Implemented *--force_ref* option to force generation of reference results, even if they already exist.
* Implemented job execution with time command. Time relevant data is written to summary and psv files.
* Added job execution error checking.  Bad exit status will result in updated error fields in summary/psv files
and status display.

#### New in Previous Release  -  [v1.2.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v1.2.0)
* Fixed a problem with the when lock_file was created and checked.  Now placed before slot initialization.
* Fixed issue where program would exit if Reference file didn't exist.  Now an error message is printed and no
comparison results are printed to summary files.
* Added command line option *--no_ref* which will not create reference results when selected.  This is useful
for characterizing potential reference WUs.
* Added color to status display.
* Modified so that status display will not show skipped jobs (Reference data already exists).
* Updated reference WUs in the *WU_test/safe* directory.  Still need a WU with a Gaussian signal.

#### New in Previous Release  -  [v1.1.0](https://github.com/Ricks-Lab/benchMT/releases/tag/v1.1.0)
* Command line options can now be specified in mode lines of the BenchCFG file.  Options given on the command line
will override modes specified in the CFG file.
* An alternative CFG file can now be specified as a command line option.
* Signal counts and Angle Range are now included in the psv and txt summary files.
* Remove app *-device N* arg if specified, since -device is automatically added based on slot assignment.
* Added *--gpu_devices x,y* command line option to specify which GPU devices the user would like to include in the
benchmark run.
* Added a lock_file in the working directory to prevent a second occurrence of benchMT from using the same directory.
* Updated reference WUs in the *WU_test/safe* directory.
* Changed *--ref_signals* option to *--std_signals* for clarity.
