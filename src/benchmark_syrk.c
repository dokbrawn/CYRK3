#include "syrk.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

typedef void (*cblas_ssyrk_t)(int, int, int, int, int, float, const float*, int, float, float*, int);
typedef void (*openblas_set_threads_t)(int);

static double get_time() {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
}

void run_benchmark(int n, int k) {
    const int runs = 10;
    int thread_counts[] = {1, 2, 4, 8, 16}; 
    int t_idx, r;
    
    float *a = (float*)malloc(n * k * sizeof(float));
    float *c = (float*)malloc(n * n * sizeof(float));
    if (!a || !c) return;

    for(r = 0; r < n * k; r++) a[r] = (float)rand() / RAND_MAX;

    HMODULE openblas_dll = LoadLibraryA("libopenblas.dll");
    cblas_ssyrk_t openblas_ssyrk = NULL;
    openblas_set_threads_t set_threads = NULL;
    
    if (openblas_dll) {
        openblas_ssyrk = (cblas_ssyrk_t)GetProcAddress(openblas_dll, "cblas_ssyrk");
        set_threads = (openblas_set_threads_t)GetProcAddress(openblas_dll, "openblas_set_num_threads");
    }

    printf("\n--- SYRK BENCHMARK (N=%d, K=%d) ---\n", n, k);
    printf("Threads | Impl | Best (s) | GeoMean (s) | GFLOPS | %% of OpenBLAS\n");
    printf("----------------------------------------------------------------\n");

    double *times = (double*)malloc(runs * sizeof(double));

    for (t_idx = 0; t_idx < 5; t_idx++) {
        int threads = thread_counts[t_idx];
        double log_sum = 0;
        double best_time = 1e9;

        for (r = 0; r < runs; r++) {
            double t0 = get_time();
            syrk_s_parallel(CblasRowMajor, CblasUpper, CblasNoTrans, n, k, 1.0f, a, k, 0.0f, c, n, threads);
            double t1 = get_time();
            times[r] = t1 - t0;
            if (times[r] < best_time) best_time = times[r];
            log_sum += log(times[r]);
        }
        double geo_mean = exp(log_sum / runs);
        double gflops = (double)n * n * k / best_time / 1e9;

        double ref_time = 0;
        if (openblas_ssyrk && set_threads) {
            set_threads(threads);
            double ref_best = 1e9;
            for(r = 0; r < runs; r++) {
                double t0 = get_time();
                __try {
                    openblas_ssyrk(101, 121, 111, n, k, 1.0f, a, k, 0.0f, c, n);
                } __except(EXCEPTION_EXECUTE_HANDLER) { }
                double t1 = get_time();
                if ((t1 - t0) < ref_best) ref_best = (t1 - t0);
            }
            ref_time = ref_best;
            printf("%7d | Custom | %8.4f | %11.4f | %6.2f | %6.2f%%\n", 
                   threads, best_time, geo_mean, gflops, (ref_time / best_time) * 100.0);
            printf("        | OBLAS  | %8.4f |      -      |   -    | 100.00%%\n", ref_time);
        } else {
            printf("%7d | Custom | %8.4f | %11.4f | %6.2f | N/A\n", 
                   threads, best_time, geo_mean, gflops);
        }
    }
    free(times);
    free(a); 
    free(c);
    if (openblas_dll) FreeLibrary(openblas_dll);
}

int main() {
    run_benchmark(4096, 4096);
    return 0;
}