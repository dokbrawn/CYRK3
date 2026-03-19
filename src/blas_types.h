#ifndef BLAS_TYPES_H
#define BLAS_TYPES_H

typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum {
    CblasUpper = 121,
    CblasLower = 122
} CBLAS_UPLO;

typedef enum {
    CblasNoTrans = 111,
    CblasTrans = 112
} CBLAS_TRANSPOSE;

#endif
