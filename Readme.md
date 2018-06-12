# gpu-normal-computation
Repository accompanying the associated Kapernikov blog post.

## Build and install
### Prerequisites
- OpenCL development files

### Instructions
The build uses cmake. Use the usual cmake incantations with an appropriate generator. Make sure to build a _Release_ build when running the benchmarks (add `-DCMAKE_BUILD_TYPE=Release` to the cmake command line). In addition, install the binaries. A custom install folder can be set by passing `-DCMAKE_INSTALL_PREFIX=<custom install dir>` to the cmake command line. The default installation directory is `/usr/local`. See below for some examples on specific platforms.

_Note_: Check the .gitlab-ci.yml file for an example system installation and configuration.

### Examples
#### Linux
The example uses the `make` generator.

```
$ cmake -H. -Bbuild -DCMAKE_INSTALL_PREFIX=install -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
$ make -C build --jobs 4
$ make -C build install
```
The required binaries will then be present in the _install_ directory.

#### Windows
The example uses the `ninja` generator.

Prepare the build environment:
1. Open the appropriate build environment shell
1. Make sure that cmake.exe and ninja.exe are in your path

Then execute:
```
$ cmake -G "Ninja" -H. -Bbuild -DCMAKE_INSTALL_PREFIX=install -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
$ ninja -C build --jobs 4
$ ninja -C build install
```
The required binaries will then be present in the _install_ directory.

## Run
### Prerequisites
- A properly functioning OpenCL runtime
- _For the benchmarks:_ A tool to fix your CPU to a specific frequency (e.g. cpupower on \*NIX systems)
- _For the benchmarks:_ A binary built in release mode

### Benchmarks
#### Linux
```
$ <installation_dir>/bin/benchmark_normal_computation
```

#### Windows
```
$ <installation_dir>\bin\benchmark_normal_computation.exe
```

### Unittests
#### Linux
```
$ <installation_dir>/bin/test_normal_computation
```

#### Windows
```
$ <installation_dir>\bin\test_normal_computation.exe
```
