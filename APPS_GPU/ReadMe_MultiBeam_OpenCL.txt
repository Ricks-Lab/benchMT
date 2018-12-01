MultiBeam OpenCL application is intended to process SETI@home MultiBeam v6,v7,v8 tasks and forthcoming "large" tasks.

Source code repository: https://setisvn.ssl.berkeley.edu/svn/branches/sah_v7_opt
Build from SVN revision: 3584
Date of revision commit: 2016/12/09 10:22:00

Available command line switches:

-v N :sets level of verbosity of app. N - integer number.  Default corresponds to -v 1. 
    -v 0 disables almost all output.
    Levels from 2 to 5 reserved for increasing verbosity, higher levels reserved for specific usage.
    -v 2 enables all signals output.
    -v 6 enables delays printing where sleep loops used.
    -v 7 enables oclFFT config printing for oclFFT fine tune.
    -v 8 prints kernel launch configuration for PulseFind algorithm

-period_iterations_num N : Splits single PulseFind kernel call to N calls for longest PulseFind calls. Can be 
    used to reduce GUI lags or to prevent driver restarts. Can affect performance. Experimentation 
    required. Default value for v6/v7/v8 task is N=20. N should be positive integer.

-pref_wg_size N : Sets preferred workgroup size for Pulsefind kernels. 
	Should be multiple of wave size (32 for nVidia, 64 for ATi) for better performance 
	and doesn't exceed maximal possible WG size for particular device (256 for ATi and Intel, less than 2048 for NV, depending on CC of device).

-pref_wg_num_per_cu N : Sets preferred number of workgroups per compute unit. Currently used only in PulseFind kernels.

-spike_fft_thresh N : Sets threshold FFT size where switch between different SpikeFind algorithms occurs.

-sbs N :Sets maximum single buffer size for GPU memory allocations. N should be positive integer and means 
    bigger size in Mbytes. Can affect performance and total memory requirements for application to run. 
    Experimentation required.

-hp : Results in bigger priority for application process (normal priority class and abothe normal thread priority). 
	Can be used to increase GPU load, experimentation required for particular GPU/CPU/GPU driver combo.

-high_prec_timer : Windows-only. Attempts to improve Windows multimedia timer resolution. May result in smaller Sleep quantum.
	In turn this would allow finer-grain sleep with less performance degradation (if any) with -use_sleep option. 

-cpu_lock : Enables CPUlock feature. Results in CPUs number limitation for particular app instance. Also attempt to bind different instances to different CPU cores will be made.
	Can be used to increase performance under some specific conditions. Can decrease performance in other cases though. Experimentation required.
	Now this option allows GPU app to use only single logical CPU. 
	Different instances will use different CPUs as long as there is enough of CPU in the system.
	Use -instances_per_device N option if multiple instances on GPU device are used.

-total_GPU_instances_num N : To use together with -cpu_lock on multi-vendor GPU hosts. Set N to total number of simultaneously running GPU
	OpenCL SETI apps for host (total among all used GPU of all vendors). App needs to know this number to properly select logical CPU for execution
	in affinity-management (-cpu_lock) mode.  Should not exceed 64.

-cpu_lock_fixed_cpu N : Will enable CPUlock too but will bind all app instances to the same N-th CPU  (N=0,1,.., number of CPUs-1).

-no_cpu_lock : To disable affinity management (opposite to -cpu_lock option). For ATi version CPUlock affinity management enabled by default.

-no_defaults_scaling : Disables auto-tuning default parameters. Basic params will be used. Implies user-supplied tuning.

-high_perf : Sets app to high-performance path. Next message will appear in stderr if option used correctly:
	"High-performance path selected. If GUI lags occur consider to remove -high_perf option from tuning line"

-low_perf : Sets app to low-performance  path. Next message will appear in stderr if option used correctly:
	"Low-performance path selected. Additional measures against GUI lags will be taken but performance can degrade"

-instances_per_device N :Sets allowed number of simultaneously executed GPU app instances per GPU device (shared with MultiBeam app instances). 
	N - integer number of allowed instances.  Should not exceed 64.

-gpu_lock :Old way GPU lock enabled. Use -instances_per_device N switch to provide number of instances to run.

These 2 options used together provide BOINC-independent way to limit number of simultaneously executing GPU 
apps. Each SETI OpenCL GPU application with these switches enabled will create/check global Mutexes and 
suspend its process execution if limit is reached. Awaiting process will consume zero CPU/GPU and rather low 
amount of memory awaiting when it can continue execution.

-use_sleep :Results in additional Sleep() calls to yield CPU to other processes. Can affect performance. Experimentation required.

-use_sleep_ex N: enables use_sleep; sets argument of Sleep() call to N: Sleep(N)

-no_use_sleep : Disables Sleep() usage. Opposite to -use_sleep. For Linux NV path use_sleep enabled by default. This option can disable it.

-sleep_quantum N: informs app what size of sleep quantum particular host has. N in ms. Some AMD-based systems have 15ms quantum.
	Some Intel-based systems have smaller one but still larger than 1ms. App will use default value of 15ms if this switch omitted.
	Has no effect in iGPU build (USE_OPENCL_INTEL path).
	Enables use_sleep.

-sleep_margin F: Number from 0.001 to 0.9999 that designates what part of sleep quantum will trigger new quantum addition to sleep interval.
	Default is 0.8 that means kernel sequence execution times of 80% of sleep quantum or more will trigger sleep at least sleep_quantum long.
	If kernel sequence execution time less than sleep_margin*sleep_quantum zero argument for Sleep() call will be used.
	To save more CPU time (for NV GPUs) at cost of GPU performance decrease this value. 
	To reduce possible GPU idle time (in case use_sleep enabled and single app instance per GPU used) at cost of CPU consumption (for NV GPUs)
	increase this param value.
	Has no effect in iGPU build (USE_OPENCL_INTEL path).
	Enables use_sleep.
	
-tt F: Sets desired target time for kernel sequence. That is, how long kernel/kernel sequence can executes w/o interruption and w/o switching
	to another tasks like GUI update. F is floating point number in milliseconds. Default is 15ms. App will try to adapt kernels (currently
	implemented for PulseFind kernels) to run designated amount of time. To increase performance try to increase this value. ?High values
	could result in GUI lags. If use_sleep active try to use target times divisible on sleeping time quantum for your particular system.
	For example at least some AMD-based systems have 15ms sleep quantum. That is, Sleep(1) will actually sleep 15ms instead of 1ms.
	Has no effect in iGPU build (USE_OPENCL_INTEL path).

-no_caching : Disables CL files binary caching
 
-tune N Mx My Mz : to make app more tunable this param allows user to fine tune kernel launch sizes of most important kernels.
	N - kernel ID (see below)
	Mxyz - workgroup size of kernel. For 1D workgroups Mx will be size of first dimension and My=Mz=1 should be 2 other ones.
	N should be one of values from this list:
	TRIPLET_HD5_WG=1, 
	For best tuning results its recommended to launch app under profiler to see how particular WG size choice affects particular kernel.
	This option mostly for developers and hardcore optimization enthusiasts wanting absolute max from their setups.
	No big changes in speed expected but if you see big positive change over default please report.
	Usage example: -tune 1 2 1 64  (set workgroup size of 128 (2x1x64) for TripletFind_HD5 kernels).


This class of options tunes oclFFT performance
-oclfft_tune_gr N : Global radix
-oclfft_tune_lr N : Local radix
-oclfft_tune_wg N : Workgroup size
-oclfft_tune_ls N : Max size of local memory FFT
-oclfft_tune_bn N : Number of local memory banks
-oclfft_tune_cw N : Memory coalesce width

For examples of app_info.xml entries look into text file with .aistub extension provided in corresponding 
package.

Command line switches can be used either in app_info.xml or mb_cmdline.txt.
Params in mb_cmdline*.txt will override switches in <cmdline> tag of app_info.xml.


For device-specific settings in multi-GPU systems it's possible to override some of command-line options via 
application config file.

Name of this config file:
MultiBeam_<vendor>_config.xml where vendor can be ATi, NV or iGPU.
File structure:
<deviceN>
	<period_iterations_num>N</period_iterations_num>
	<spike_fft_thresh>N</spike_fft_thresh>
	<sbs>N</sbs>
	<oclfft_plan>
	        <size>N</size>
		<global_radix>N</global_radix>
		<local_radix>N</local_radix>
		<workgroup_size>N</workgroup_size>
		<max_local_size>N</max_local_size>
		<localmem_banks>N</localmem_banks>
		<localmem_coalesce_width>N</localmem_coalesce_width>
	</oclfft_plan>
	<no_caching>
</deviceN>
where deviceN - particular OpenCL device N starting with 0, multiple sections allowed, one per each device.
other fields - corresponding command-line options to override for this particular device.
All or some sections can be omitted.

Don't forget to re-check device number to physical card relation after driver update and physical slot change. 
Both these actions can change cards enumeration order.

Best usage tips:
For best performance it is important to free 2 CPU cores running multiple instances.
Freeing at least 1 CPU core is necessity to get enough GPU usage.

If you experience screen lags or driver restarts increase -period_iteration_num in app_info.xml or mb_cmdline*.txt
It is more important to free CPU core(s) so more instances are running. 


============= ATi specific info ===============
Known issues:

  With 12.x Catalyst drivers GPU usage can be low if CPU fully used with another loads. App performance can be 
  increased by using -cpu_lock switch in this case.

- Catalyst 12.11 beta and 13.1 have broken OpenCL compiler that will result in driver restarts or invalid 
  results. But these drivers can be used still if the kernels binaries are precompiled under an older Catalyst 
  driver. That is, delete all *.bin* files from SETI project directory, revert to Catalyst 12.8 or 12.10, or 
  upgrade to Catalyst 13.2 or later, process at least one task (check that those *.bin* files were generated 
  again) and (if needed) update to Catalyst 13.1. New builds with versioned binary cachase will require 
  additional step: to rename old bin_* files to match name (driver version part of name) of newly generated 
  ones.

App instances:

On high end cards HD 5850/5870, 6950/6970, 7950/7970, R9 280X/290X running 3 instances should be fastest.
HD 7950/7970 and R9 280X/290X can handle 4 instances very well. Testing required.
Beware free CPU cores.

On mid range cards HD 5770, 6850/6870, 7850/7870 best performance should be running 2 instances.

If you experience screen lags or driver restarts increase -period_iteration_num in app_info.xml or mb_cmdline*.txt
It is more important to free CPU core(s) so more instances are running. 

command line switches.
______________________

Running 3 instances set -sbs 192 is best option for speed on 1GB GPU.
If only 2 instances are running set -sbs 256 for max speed up.
Users using GPU with 3 GB RAM set -sbs 244 - 280 for best speed also running 3 or more instances.
This might require some fine tuning.

One instance requires aprox. 500 MB VRAM (depends on -sbs value used)

Entry Level cards HD x3xx / x4xx R7 230/240/250
-spike_fft_thresh 2048 -tune 1 2 1 16

Mid range cards x5xx / x6xx / x7xx / R9 260 / 270
-spike_fft_thresh 2048 -tune 1 64 1 4

High end cards x8xx / x9xx / R9 280x / 290x
-spike_fft_thresh 4096 -tune 1 64 1 4 -oclfft_tune_gr 256 -oclfft_tune_lr 16 -oclfft_tune_wg 256 -oclfft_tune_ls 512 -oclfft_tune_bn 64 -oclfft_tune_cw 64

============= Intel specific info =============

Suggested command line switches:

HD 2500 
 -spike_fft_thresh 2048 -tune 1 2 1 16 (*requires testing)

HD 4000
 -spike_fft_thresh 2048 -tune 1 64 1 4 (*requires testing)

HD 4200 / HD 4600 / HD 5xxx
 -spike_fft_thresh 4096 -tune 1 64 1 4 -oclfft_tune_gr 256 -oclfft_tune_lr 16 -oclfft_tune_wg 512 (*requires testing)


============= NV specific info ================
Known issues:

- With NV drivers past 267.xx GPU usage can be low if CPU fully used with another loads. App performance can 
  be increased by using -cpu_lock switch in this case, and CPU time savings are possible with -use_sleep switch.

Suggested command line switches:

Entry Level cards NV x20 x30 x40
 -sbs 128 -spike_fft_thresh 2048 -tune 1 2 1 16 (*requires testing)

Mid range cards x50 x60 x70
 -sbs 192 -spike_fft_thresh 2048 -tune 1 64 1 4 (*requires testing)

High end cards x8x 780TI Titan / Titan Z
 -sbs 256 -spike_fft_thresh 4096 -tune 1 64 1 4 -oclfft_tune_gr 256 -oclfft_tune_lr 16 -oclfft_tune_wg 256 -oclfft_tune_ls 512 -oclfft_tune_bn 64 -oclfft_tune_cw 64 (*requires testing)

===============================================
