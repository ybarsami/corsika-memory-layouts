# corsika-memory-layouts

Some benchmark code in C related to the CORSIKA code, see
https://gitlab.ikp.kit.edu/AirShowerPhysics/corsika/issues/224

The main idea is to test the efficiency of vectorized operations (Array of
Structures VS Structure of Arrays) on a memory-intensive benchmark.


## Source code

See `memory_layouts_benchmarks_func.c`

This is an adaptation from the benchmark of Hans Dembinski that can be found at
the following address: https://github.com/HDembinski/corsika_span_demo


## Compile the code

The code has been compiled and tested with the Intel C Compiler:

icc (ICC) 19.0.5.281 20190815

`icc memory_layouts_benchmarks_func.c -lm -O3 -qopenmp -march=native -std=gnu11 -Wall -o test.out -qopt-report=5 -qopt-report-phase=vec`

It also compiles with the GNU C Compiler, but the efficiency results when using
gcc cannot really be used, because to vectorize the `logf()` function call, we
need to compile with `-ffast-math`, and the resulting code is less efficient on
Intel hardware.


## Vectorization report

See `memory_layouts_benchmarks_func.optrpt`

All the AoS loops are vectorized with stride 9, all the SoA loops are
vectorized with stride 1.


## Overall efficiency

The code here is memory bound, and should not be too dependent on the most
efficient vectorization. Indeed, here, it looks like the two versions that run
faster are the AOS_1 (line 292) and SOA_2A (line 380).

It would be very interesting to test with a more realistic scenario where more
operations are performed on particles, and where not all of them are
vectorizable (as is usually the case).

