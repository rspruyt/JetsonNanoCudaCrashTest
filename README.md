# 16 vs 8-bit kernel surface read test
Demonstrating 8-bit surface reading works fine in a kernel but 16-bit seems to cause the kernel to fail

Enable or disable #define USE_16_BIT to demonstrate the issue

e.g. (works fine using 8-bit surface/data):

```
make
./TestCuda
```

or e.g. (kernel fails using 16-bit surface/data):

```
make NVCCFLAGS=-DUSE_16_BIT
./TestCuda
```
