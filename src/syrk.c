#include "syrk.h"
#include <stddef.h>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

#define BLOCK_SIZE 64

int syrk_validate(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                  int n, int k, int lda, int ldc) {
    if (n < 0 || k < 0) return 0;
    if (order != CblasRowMajor && order != CblasColMajor) return 0;
    if (uplo != CblasUpper && uplo != CblasLower) return 0;
    if (trans != CblasNoTrans && trans != CblasTrans) return 0;

    int req_lda;
    if (order == CblasColMajor) {
        req_lda = (trans == CblasNoTrans) ? k : n;   
    } else {
        req_lda = (trans == CblasNoTrans) ? k : n;
    }

    if (lda < (req_lda > 0 ? req_lda : 1)) return 0;
    if (ldc < (n > 0 ? n : 1)) return 0;

    return 1;
}


static void syrk_internal_s(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                           int n, int k, float alpha, const float *a, int lda,
                           float beta, float *c, int ldc, int use_omp) {
    int i, j, p, ii, jj, pp;

    #pragma omp parallel for private(i, j) if(use_omp)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                int c_idx = (order == CblasRowMajor) ? (i * ldc + j) : (i + j * ldc);
                c[c_idx] = (beta == 0.0f) ? 0.0f : c[c_idx] * beta;
            }
        }
    }

    if (alpha == 0.0f) return;

    if (order == CblasRowMajor && trans == CblasNoTrans) {
        #pragma omp parallel for private(ii, jj, pp, i, j, p) schedule(dynamic) if(use_omp)
        for (ii = 0; ii < n; ii += BLOCK_SIZE) {
            for (jj = 0; jj < n; jj += BLOCK_SIZE) {
                if (uplo == CblasUpper && ii > jj + BLOCK_SIZE) continue;
                if (uplo == CblasLower && jj > ii + BLOCK_SIZE) continue;
                for (pp = 0; pp < k; pp += BLOCK_SIZE) {
                    for (i = ii; i < ii + BLOCK_SIZE && i < n; i++) {
                        int j_start = (uplo == CblasUpper) ? (i > jj ? i : jj) : jj;
                        int j_end = (uplo == CblasLower) ? (i < jj + BLOCK_SIZE ? i + 1 : jj + BLOCK_SIZE) : (jj + BLOCK_SIZE);
                        if (j_end > n) j_end = n;
                        for (j = j_start; j < j_end; j++) {
                            float sum = 0.0f;
                            const float *a_i = &a[i * lda + pp];
                            const float *a_j = &a[j * lda + pp];
                            int k_lim = (pp + BLOCK_SIZE > k) ? (k - pp) : BLOCK_SIZE;
                            #pragma omp simd reduction(+:sum)
                            for (p = 0; p < k_lim; p++) { sum += a_i[p] * a_j[p]; }
                            c[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for private(i, j, p) schedule(dynamic) if(use_omp)
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                    float sum = 0.0f;
                    for (p = 0; p < k; p++) {
                        float v1, v2;
                        if (order == CblasRowMajor) {
                            v1 = (trans == CblasNoTrans) ? a[i * lda + p] : a[p * lda + i];
                            v2 = (trans == CblasNoTrans) ? a[j * lda + p] : a[p * lda + j];
                        } else {
                            v1 = (trans == CblasNoTrans) ? a[i + p * lda] : a[p + i * lda];
                            v2 = (trans == CblasNoTrans) ? a[j + p * lda] : a[p + j * lda];
                        }
                        sum += v1 * v2;
                    }
                    int c_idx = (order == CblasRowMajor) ? (i * ldc + j) : (i + j * ldc);
                    c[c_idx] += alpha * sum;
                }
            }
        }
    }
}


static void syrk_internal_d(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                           int n, int k, double alpha, const double *a, int lda,
                           double beta, double *c, int ldc, int use_omp) {
    int i, j, p, ii, jj, pp;

    #pragma omp parallel for private(i, j) if(use_omp)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                int c_idx = (order == CblasRowMajor) ? (i * ldc + j) : (i + j * ldc);
                c[c_idx] = (beta == 0.0) ? 0.0 : c[c_idx] * beta;
            }
        }
    }

    if (alpha == 0.0) return;

    if (order == CblasRowMajor && trans == CblasNoTrans) {
        #pragma omp parallel for private(ii, jj, pp, i, j, p) schedule(dynamic) if(use_omp)
        for (ii = 0; ii < n; ii += BLOCK_SIZE) {
            for (jj = 0; jj < n; jj += BLOCK_SIZE) {
                if (uplo == CblasUpper && ii > jj + BLOCK_SIZE) continue;
                if (uplo == CblasLower && jj > ii + BLOCK_SIZE) continue;
                for (pp = 0; pp < k; pp += BLOCK_SIZE) {
                    for (i = ii; i < ii + BLOCK_SIZE && i < n; i++) {
                        int j_s = (uplo == CblasUpper) ? (i > jj ? i : jj) : jj;
                        int j_e = (uplo == CblasLower) ? (i < jj + BLOCK_SIZE ? i + 1 : jj + BLOCK_SIZE) : (jj + BLOCK_SIZE);
                        if (j_e > n) j_e = n;
                        for (j = j_s; j < j_e; j++) {
                            double sum = 0.0;
                            const double *a_i = &a[i * lda + pp];
                            const double *a_j = &a[j * lda + pp];
                            int k_l = (pp + BLOCK_SIZE > k) ? (k - pp) : BLOCK_SIZE;
                            #pragma omp simd reduction(+:sum)
                            for (p = 0; p < k_l; p++) { sum += a_i[p] * a_j[p]; }
                            c[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for private(i, j, p) schedule(dynamic) if(use_omp)
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                    double sum = 0.0;
                    for (p = 0; p < k; p++) {
                        double v1, v2;
                        if (order == CblasRowMajor) {
                            v1 = (trans == CblasNoTrans) ? a[i * lda + p] : a[p * lda + i];
                            v2 = (trans == CblasNoTrans) ? a[j * lda + p] : a[p * lda + j];
                        } else {
                            v1 = (trans == CblasNoTrans) ? a[i + p * lda] : a[p + i * lda];
                            v2 = (trans == CblasNoTrans) ? a[j + p * lda] : a[p + j * lda];
                        }
                        sum += v1 * v2;
                    }
                    int c_idx = (order == CblasRowMajor) ? (i * ldc + j) : (i + j * ldc);
                    c[c_idx] += alpha * sum;
                }
            }
        }
    }
}


void syrk_s(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, float alpha, const float *a, int lda, float beta, float *c, int ldc) {
    if (!syrk_validate(order, uplo, trans, n, k, lda, ldc)) return;
    syrk_internal_s(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, 0);
}

void syrk_d(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, double alpha, const double *a, int lda, double beta, double *c, int ldc) {
    if (!syrk_validate(order, uplo, trans, n, k, lda, ldc)) return;
    syrk_internal_d(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, 0);
}

void syrk_s_parallel(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, float alpha, const float *a, int lda, float beta, float *c, int ldc, int num_threads) {
    if (!syrk_validate(order, uplo, trans, n, k, lda, ldc)) return;
#ifdef HAS_OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#endif
    syrk_internal_s(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, 1);
}

void syrk_d_parallel(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, double alpha, const double *a, int lda, double beta, double *c, int ldc, int num_threads) {
    if (!syrk_validate(order, uplo, trans, n, k, lda, ldc)) return;
#ifdef HAS_OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#endif
    syrk_internal_d(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc, 1);
}