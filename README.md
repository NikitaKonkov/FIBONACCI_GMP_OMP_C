# FIBONACCI_GMP_OMP_C

>A blazing-fast, accurate Fibonacci number calculator using the GMP (GNU Multiple Precision Arithmetic) library for arbitrary-precision integer math with optional OpenMP parallelization*.


## Features
- Computes F(N) for any N (default: 20,000,000)
- Uses GMP for true arbitrary-precision (no overflow, no floating point)
- Extremely fast: O(log n * M(d)), where M(d) is the multiplication time for d digits
- OpenMP parallelization support for ~1.5-2x speedup on multi-core systems
- Automatic fallback to single-threaded when OpenMP unavailable
- Save results to file with a flag

## Requirements
- GCC (or compatible C compiler)
- GMP library (https://gmplib.org/)

## Build

```sh
# Simple build (reliable & fast):
windows:
    gcc -O3 -march=native -mtune=native -fopenmp -o fib.exe fibonacci.c -lgmp
linux:
    gcc -O3 -march=native -mtune=native -fopenmp -o fib fibonacci.c -lgmp
android:
    pkg install gmp clang openmp
    clang -O3 -march=native -mtune=native -fopenmp -o fib fibonacci.c -lgmp

# Maximum performance build (aggressive optimizations):
windows:
    gcc -O3 -march=native -mtune=native -mavx512f -fopenmp -flto -funroll-loops -finline-functions -fomit-frame-pointer -falign-functions=32 -falign-loops=32 -fprefetch-loop-arrays -fno-stack-protector -DNDEBUG -o fib.exe fibonacci.c -lgmp
linux:
    gcc -O3 -march=native -mtune=native -mavx512f -fopenmp -flto -funroll-loops -finline-functions -fomit-frame-pointer -falign-functions=32 -falign-loops=32 -fprefetch-loop-arrays -fno-stack-protector -DNDEBUG -o fib fibonacci.c -lgmp
android:
    clang -O3 -march=native -mtune=native -fopenmp -flto -funroll-loops -finline-functions -fomit-frame-pointer -falign-functions=32 -falign-loops=32 -fprefetch-loop-arrays -fno-stack-protector -DNDEBUG -o fib fibonacci.c -lgmp

# Note: OpenMP (-fopenmp) is optional. Without it, runs single-threaded.
```

## Usage

```sh
./fib.exe [-s] [-h] [N]
```
- `-s` : Save result to file (optional)
- `-h` : Show help message
- `N`  : Fibonacci number to compute

## Algorithm Deep Dive: Fast Doubling Recursion

The algorithm uses a **divide-and-conquer** approach that recursively splits the problem in half until reaching the base case, then computes results as the recursion unwinds:

### Recursion Flow
```
fast_fib_calc(1000000)
├─ fast_fib_calc(500000)
   ├─ fast_fib_calc(250000)
      └─ ... keeps dividing by 2 (≈20 levels deep)
         └─ fast_fib_calc(0) ← BASE CASE: returns A=0, B=1
```

As recursion unwinds, each level computes F(2k) and F(2k+1) from F(k) and F(k+1) using fast doubling formulas, with OpenMP parallelization for N ≥ 100M.

**Why it's fast**: O(log N) levels × O(M(d)) per level = **O(log N × M(d))** instead of O(N) for naive iteration.

## Algorithm Visualization

The program uses a recursive "fast doubling" approach with OpenMP parallelization for large N:

```
                    fast_fib_calc(N, A, B)
                            ↓
                    ┌─────────────────────┐
                    │  A = F(N)           │  ← Output (mpz_t)
                    │  B = F(N+1)         │  ← Big integers
                    └─────────────────────┘
                            ↑
              ┌─────────────┴─────────────┐
              │                           │
        if N == 0:                      else:
      A = 0, B = 1                   Initialize: F0, F1, TEMP, F0_Q
        (base case)                         ↓
           return                    fast_fib_calc(N/2, F0, F1)
                                            ↓
                          ┌─────────────────────────────┐
                          │ F0 = F(N/2)                 │
                          │ F1 = F(N/2+1)               │
                          └─────────────────────────────┘
                                          ↓
                    Apply Fast Doubling Formulas (Parallel if N >= 100K):
                    ┌──────────────────────┬──────────────────────┐
                    │  #pragma omp section │  #pragma omp section │
                    │ ┌──────────────────┐ │ ┌──────────────────┐ │
                    │ │ TEMP = F1 << 1   │ │ │ B = F1 * F1      │ │
                    │ │ TEMP = TEMP - F0 │ │ │ F0_Q = F0 * F0   │ │
                    │ │ A = F0 * TEMP    │ │ │ B = B + F0_Q     │ │
                    │ │ // F(2k)         │ │ │ // F(2k+1)       │ │
                    │ └──────────────────┘ │ └──────────────────┘ │
                    └──────────────────────┴──────────────────────┘
                                          ↓
                                   ┌─────────────┐
                                   │  N % 2 != 0?│
                                   │  (N is odd) │
                                   └─────────────┘
                                      ↙       ↘
                                 No ↙           ↘ Yes
                            ┌──────────┐     ┌───────────────┐
                            │ Keep:    │     │ mpz_swap(A,B) │
                            │ A=F(2k)  │     │ mpz_add(B,B,A)│
                            │ B=F(2k+1)│     │ → A=F(N)      │
                            └──────────┘     │   B=F(N+1)    │
                                             └───────────────┘
                                          ↓
                               Clear: F0, F1, TEMP, F0_Q
                                    return A, B

Recursion Call Stack for N=13:
    fast_fib_calc(13)      // N odd
    ├─ fast_fib_calc(6)    // 13/2 = 6
       ├─ fast_fib_calc(3)  // 6/2 = 3, N odd  
          ├─ fast_fib_calc(1) // 3/2 = 1, N odd
             └─ fast_fib_calc(0) ← Base: A=0, B=1
             └─ Compute F(2), F(3), then swap+add → F(1)=1, F(2)=1
          └─ Compute F(6), F(7), then swap+add → F(3)=2, F(4)=3  
       └─ Compute F(12), F(13) → F(6)=8, F(7)=13
    └─ Compute F(26), F(27), then swap+add → F(13)=233, F(14)=377

Time Complexity: O(log N) levels × O(M(d)) per level = O(log N × M(d))
Space Complexity: O(d) where d = digits in F(N), O(log N) stack depth
```

## Examples

```sh
$ ./fib.exe -s 16
Computing F(16)...
Result will be saved to file.
Computation completed in 0.000056 seconds

F(16) has 4 digits
Number saved to: Fibonacci_16.txt
Full number: 987
Performance: 71301 digits/second



$ ./fib.exe -h
Usage: ./fib.exe [-s] [-h] [N]
  -s            Save result to file 
  -h            Show this help message
  N             Fibonacci number to compute

Best Examples with N=16:
  ./fib.exe 16          # Output: Computing F(16)... F(16) = 987
  ./fib.exe -s 16       # Output: Computing F(16)... saves to Fibonacci_16.txt
  ./fib.exe -h          # Show this help message
```

## Performance Benchmarks

**F(1,000,000,000) - World-Class Performance:**

```sh
# ARM 3GHz CPU
Computation completed in 10.168324 seconds
Performance: 20552812 digits/second

# x86 3GHz CPU
Computation completed in 7.216904 seconds
Performance: 28958076 digits/second

# x86 5GHz CPU
Computation completed in 4.577705 seconds
Performance: 45653362 digits/second
```
*F(1 billion) computed in under 5 seconds - good performance for a cpu only approach*

## License
MIT
