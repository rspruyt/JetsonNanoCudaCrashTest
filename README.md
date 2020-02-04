#JetsonNanoCudaCrashTest
Demonstrating a Jetson Nano Cuda Crash when running an unreleated kernel and allocation in seperate threads.

Enable or disable #define USE_MUTEX to demonstrate the issue

e.g. (works fine):

make NVCCFLAGS=-DUSE_MUTEX
./TestCuda
or e.g. (crashes):

make
./TestCuda
