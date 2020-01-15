# JetsonNanoCudaCrashTest
Demonstrating a Jetson Nano Cuda Crash when running an unreleated kernel and allocation in seperate threads. 

Enable or disable #define USE_MUTEX to demonstrate the issue

e.g. (works fine):
```
make all NVCCFLAGS=-DUSE_MUTEX
```

or e.g. (crashes):
```
make all
```
