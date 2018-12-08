# benchMT  -  SETI multi-threaded MB Benchmark Tool

 This tool will extract the total number of CPU cores/threads and GPU platforms from the user's
 environment and utilize them in running a list of apps/args specified in the benchCFG file.  Using
 less than total number of CPU threads can be specified in the command line.  This tool will read a
 list of MB apps/args from the benchCFG file and search for the specified MB apps in the APP_CPU
 and APP_GPU directories to validate and determine platform.  It will then leverage allocated
 threads, as specified, to run all benchmark jobs, storing results in the testData directory.  Use
 the *--help* option to get a description of valid command line arguments.

 By default, a summary list of all jobs will update in the display as the program progresses.  If
 there are a large number of jobs, then this display may not be useful and the --display_slots
 option can be used to display the status of each slot as the program progresses.  In some cases,
 there will be too many slots to display, and the --display_compact option can used to further
 optimize the progress display.

 You may need to use the *--boinc_home* command option to specify the boinc home directory, which
 is required, since boinccmd is used.

 All WUs in the WU_test directory will be used in the creation of jobs to be run, unless the 
 *--ref_signals* option is used, in which case, WUs in the WU_std_signal will be used.  The
 APPS_GPU and APPS_CPU can have more apps than are specified to run in the BenchCFG file, but must
 contain apps specified in BenchCFG.  The APPS_REF must contain a single CPU reference app with a
 file prefix of "ref-cpu.".  The stock CPU app is suggested, as this is only used to test
 integrity of the results.  Elapsed time analysis is expected to be limited to apps/arg
 combinations specified in BenchCFG.

 The results will be stored in a unique subdir of the testData directory. There is an overall run
 log txt file, a .psv file useful for importing into an analytics tools, and the .sah and stderr
 files for each job run. A run name can be specified with the *--run_name* commane line option. This
 name will be included in the name of the testData subdirectory for the current run.

## New in this Release  -  V1.1.0
* Command line options can now be specified mode lines of the BenchCFG file.  Options given on the command line will override modes specified in the CFG file.
* An alternative CFG file can now be specified as a command line option.
* Signal counts and Angle Range are now included in the psv and txt summary files.
* Remove -device arg if specified, since -device is automatically added based on slot assignment.
* Added -gpu_devices x,y command line option to specify which GPU devices the user would like to include in the benchmark run.
* Added a lock_file in the working directory to prevent a second occurrence of benchMT from using the same directory.

## Development Plans and Known Limitations
* GPU multi-threaded implementation. Currently total_gpu_threads = total_gpu_count, a future development opportunity is to implement a max number of threads per GPU
* Consider using opencl instead of lshw to get valid GPU compute platforms, but maybe won't work for cuda apps
* Should consider executing job with time command.  This should give total and CPU time metrics
* Need to figure out how to run a job without a shell
* Deal with an immediate fail to spawn a process when executing a job

