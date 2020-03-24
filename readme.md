## Matrix Invert

---------------------------

### Introduction

Optimization of inverse matrix for OpenCV Gauss-Jordan method

Unit testing with Google Test，use Google Benchmark for performance comparison test

Because OpenCV did parallel optimization of the instruction set, I extracted the core code for comparison under single-core batch processing.

-----------------------------

### Installation

First you need to install Google Test and Google Benchmark

* [Google Test](https://github.com/google/googletest)
* [Google Benchmark](https://github.com/google/benchmark)

Then

```
git clone https://github.com/Lancelot0902/matrix_invert

cd matrix_invert

mkdir build && cd build

cmake ..

make
```

### Usage

```
./invert_unittest

./invert_benchmark
```

### TODO

Using OpenMP for parallel optimization, compared with invert under OpenCV framework
