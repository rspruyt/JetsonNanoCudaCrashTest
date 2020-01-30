# JetsonNanoCudaCrashTest
Demonstrating a Jetson Nano Cuda Crash when running an unreleated kernel and allocation in seperate threads. This was last tested on Jetpack R32.3.1 (2019/12/17 release), but the behaviour is consistent on the previous Nano release and on the TX2 with Jetpack 32.2.X.

The issue is discussed here:
https://devtalk.nvidia.com/default/topic/1069859/jetson-nano/simple-cuda-multi-threading-crash-on-nano/

** SOLVED ** Concurrent Managed Memory access while kernels are executing is not supported on Windows or Mobile/ARM64 devices as mentioned in the above thread

Enable or disable #define USE_MUTEX to demonstrate the issue

e.g. (works fine):
```
make NVCCFLAGS=-DUSE_MUTEX
./TestCuda
```

or e.g. (crashes):
```
make
./TestCuda
```
