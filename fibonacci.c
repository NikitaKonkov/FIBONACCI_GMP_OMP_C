/*
Fibonacci Fast Doubling BIG_INTEGER:
- Computes F(N) using the fast doubling method and GMP for arbitrary precision.
- Fast doubling: Recursively computes F(2k) and F(2k+1) from F(k), F(k+1) in O(log N) time.
- No approximations or floating point; exact results for huge N.
- Supports OpenMP parallelization for multi-core acceleration on large N (>1M).

Performance:
- Time: O(log N * M(d)), where M(d) is the time to multiply d-digit numbers (GMP-optimized)
- Space: O(d), d = number of digits in F(N)
- Parallel speedup: ~1.5-2x on multi-core systems for very large N

Usage:
- ./fib.exe [-s] [-h] [N]
- -s: Save result to file (optional)
- -h: Show help message
- N: Fibonacci number to compute (default: 20000000)

Compile:
    windows:
        gcc -O3 -march=native -mtune=native -fopenmp -flto -ffast-math -o fib.exe fibonacci.c -lgmp
    linux:
        gcc -O3 -march=native -mtune=native -fopenmp -flto -ffast-math -o fib fibonacci.c -lgmp

    Note: OpenMP (-fopenmp) is optional. Without it, runs single-threaded.

Info:
    This code is so portable and efficient, it'll run your smartfridge.
*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <gmp.h>

int RETURN_FLAG = 0;



// F(2k) = F(k) * (2*F(k+1) - F(k))
// F(2k+1) = F(k+1)^2 + F(k)^2
void fast_fib_calc(unsigned long N, mpz_t A, mpz_t B)
{

    // If N is 0, start compute cases
    if (N == 0) 
    {
        mpz_set_ui(A, 0);
        mpz_set_ui(B, 1);
        return;
    }

    // Init vars
    mpz_t F0, F1, TEMP, F0_Q;
    mpz_init(F0);
    mpz_init(F1);
    mpz_init(TEMP);
    mpz_init(F0_Q);

    // Recursive until N == 0
    fast_fib_calc(N / 2, F0, F1); 


    // Independent calculations when (N >= 50000000) res ~0.4 seconds
    #pragma omp parallel sections if (N >= 50000000)
    {
        #pragma omp section
        {
            // TEMP = F1 * 2^1 = F1 << 1
            mpz_mul_2exp(TEMP, F1, 1);
            // TEMP = TEMP - F0
            mpz_sub(TEMP, TEMP, F0);
            // A = F0 * TEMP
            mpz_mul(A, F0, TEMP);
        }
        #pragma omp section
        {
            // B = F1 * F1
            mpz_mul(B, F1, F1);
            // F0_Q = F0 * F0
            mpz_mul(F0_Q, F0, F0);
            // B = B + F0_Q
            mpz_add(B, B, F0_Q);
        }
    }

    // If odd do swap
    if (N % 2 != 0)
    {
        mpz_swap(A, B);
        mpz_add(B, B, A);
    }

    // Clear temp vars
    mpz_clear(F0);
    mpz_clear(F1);
    mpz_clear(TEMP);
    mpz_clear(F0_Q);
}



// Print Fibonacci number
void print_fib_info(mpz_t A, unsigned long N, int SAVE)
{
    // Basic info
    size_t NUM_DIGITS = mpz_sizeinbase(A, 10);
    printf("F(%lu) has %zu digits\n", N, NUM_DIGITS);

    // For very large nums
    if (NUM_DIGITS > 1000000 && !SAVE)
    {
        mpz_t TEMP, D;
        mpz_init(TEMP);
        mpz_init(D);
        
        mpz_ui_pow_ui(D, 10, NUM_DIGITS - 50);
        mpz_div(TEMP, A, D);
        char *FIRST_DIGITS = mpz_get_str(NULL, 10, TEMP);
        
        // Last 50 digits: A % 10^50
        mpz_ui_pow_ui(D, 10, 50);
        mpz_mod(TEMP, A, D);
        char *LAST_DIGITS = mpz_get_str(NULL, 10, TEMP);
        
        printf("First ~50 digits: %s\n", FIRST_DIGITS);
        printf("Last 50 digits:   %50s\n", LAST_DIGITS);
        
        free(FIRST_DIGITS);
        free(LAST_DIGITS);
        mpz_clear(TEMP);
        mpz_clear(D);
        return;
    }

    char *FIB_STR = NULL;
    
    // Only convert to string if needed
    if (SAVE || NUM_DIGITS <= 100 || NUM_DIGITS <= 1000000)
    {
        fflush(stdout);
        FIB_STR = mpz_get_str(NULL, 10, A);
    }

    // Save Fibonacci number to file with [-s]
    if (SAVE && FIB_STR)
    {
        char FILENAME[50];
        sprintf(FILENAME, "Fibonacci_%lu.txt", N);
        
        printf("Writing to file %s... ", FILENAME);
        fflush(stdout);
        
        FILE *FILE_PTR = fopen(FILENAME, "w");
        if (FILE_PTR != NULL)
        {
            fprintf(FILE_PTR, "F(%lu) = %s\n", N, FIB_STR);
            fclose(FILE_PTR);
            printf("Number saved to: %s\n", FILENAME);
        }
        else
        {
            printf("failed.\n");
            printf("Couldn't create file %s\n", FILENAME);
        }
    }

    // Display digits
    if (FIB_STR)
    {
        if (NUM_DIGITS > 100)
        {
            printf("First 50 digits: %.50s\n", FIB_STR);
            printf("Last 50 digits:  %s\n", FIB_STR + NUM_DIGITS - 50);
        }
        else
        {
            printf("Full number: %s\n", FIB_STR);
        }
        free(FIB_STR);
    }
}



void print_usage(const char *PROGRAM_NAME)
{
    printf("Usage: %s [-s] [-h] [N]\n", PROGRAM_NAME);
    printf("  -s             Save result to file (optional)\n");
    printf("  -h             Show this help message\n");
    printf("  N              Fibonacci number to compute (default: 20000000)\n");
    printf("\nExamples:\n");
    printf("  %s             # Compute F(20000000), don't save\n", PROGRAM_NAME);
    printf("  %s -s          # Compute F(20000000), save to file\n", PROGRAM_NAME);
    printf("  %s -s 1000000  # Compute F(1000000), save to file\n", PROGRAM_NAME);
    printf("  %s 100         # Compute F(100), don't save\n", PROGRAM_NAME);
}



int main(int argc, char *argv[])
{

    // Init GMP BigInt Arrays
    mpz_t FIB_N, FIB_N1;
    mpz_init(FIB_N);
    mpz_init(FIB_N1);

    unsigned long TARGET = 20000000; // Default: compute F(20,000,000)
    int SAVE_TO_FILE = 0;            // Default: don't save to file

    // Parse command line arguments
    for (int I = 1; I < argc; I++)
    {
        if (strcmp(argv[I], "-s") == 0)
        {
            SAVE_TO_FILE = 1; // File saving
        }
        else if (strcmp(argv[I], "-h") == 0)
        {
            print_usage(argv[0]); // help and exit

            goto EXIT;
        }
        else
        {
            char *ENDPTR;
            unsigned long PARSED = strtoul(argv[I], &ENDPTR, 10);
            if (*ENDPTR == '\0' && PARSED > 0)
            {
                TARGET = PARSED;
            }
            else
            {
                printf("Error: Invalid argument '%s'\n\n", argv[I]);
                print_usage(argv[0]);

                RETURN_FLAG = 1;
                goto EXIT;
            }
        }
    }

    printf("Computing F(%lu)...\n\n", TARGET);
    if (SAVE_TO_FILE)
    {
        printf("Result will be saved to file.\n");
    }

    struct timespec START_TIME, END_TIME;
    clock_gettime(CLOCK_MONOTONIC, &START_TIME);


    // Compute F(TARGET)
    fast_fib_calc(TARGET, FIB_N, FIB_N1); 


    clock_gettime(CLOCK_MONOTONIC, &END_TIME);
    double ELAPSED = (END_TIME.tv_sec - START_TIME.tv_sec) +
                     (END_TIME.tv_nsec - START_TIME.tv_nsec) / 1000000000.0;

    printf("Computation completed in %.6f seconds\n", ELAPSED);
    struct timespec PRINT_START, PRINT_END;
    clock_gettime(CLOCK_MONOTONIC, &PRINT_START);



    // Print results
    print_fib_info(FIB_N, TARGET, SAVE_TO_FILE); 



    clock_gettime(CLOCK_MONOTONIC, &PRINT_END);
    double PRINT_ELAPSED = (PRINT_END.tv_sec - PRINT_START.tv_sec) +
                          (PRINT_END.tv_nsec - PRINT_START.tv_nsec) / 1000000000.0;

    size_t NUM_DIGITS = mpz_sizeinbase(FIB_N, 10);
    printf("\nPerformance summary:\n");
        printf("  Computation: %.6f seconds (%.0f digits/second)\n", ELAPSED, NUM_DIGITS / ELAPSED);
    if (PRINT_ELAPSED > 0.001)
        printf("  System(I/O): %.6f seconds\n", PRINT_ELAPSED);


    EXIT:

    // Clean up GMP vars
    mpz_clear(FIB_N);
    mpz_clear(FIB_N1);

    return RETURN_FLAG;
}