#ifndef SYRK_H
#define SYRK_H

#include "blas_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int syrk_validate(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                  int n, int k, int lda, int ldc);

void syrk_s(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
            int n, int k, float alpha, const float *a, int lda,
            float beta, float *c, int ldc);

void syrk_d(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
            int n, int k, double alpha, const double *a, int lda,
            double beta, double *c, int ldc);

void syrk_s_parallel(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                     int n, int k, float alpha, const float *a, int lda,
                     float beta, float *c, int ldc, int num_threads);

void syrk_d_parallel(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                     int n, int k, double alpha, const double *a, int lda,
                     double beta, double *c, int ldc, int num_threads);

#ifdef __cplusplus
}
#endif

#endif