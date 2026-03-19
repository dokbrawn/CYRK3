#include "syrk.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#define EPS_F 1e-5f
#define EPS_D 1e-10
static int failures = 0;
static int total = 0;
static int nearly_equal_f(float a, float b) {
return fabsf(a - b) <= EPS_F;
}
static int nearly_equal_d(double a, double b) {
return fabs(a - b) <= EPS_D;
}
static int index_of(CBLAS_ORDER order, int ld, int row, int col) {
return (order == CblasColMajor) ? (row + col * ld) : (row * ld + col);
}
static int triangle_contains(CBLAS_UPLO uplo, int row, int col) {
return (uplo == CblasUpper) ? (row <= col) : (row >= col);
}
static float ref_load_f(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, const float *a, int lda, int row, int col) {
if (order == CblasColMajor) {
return (trans == CblasNoTrans) ? a[row + col * lda] : a[col + row * lda];
}
return (trans == CblasNoTrans) ? a[row * lda + col] : a[col * lda + row];
}
static double ref_load_d(CBLAS_ORDER order, CBLAS_TRANSPOSE trans, const double *a, int lda, int row, int col) {
if (order == CblasColMajor) {
return (trans == CblasNoTrans) ? a[row + col * lda] : a[col + row * lda];
}
return (trans == CblasNoTrans) ? a[row * lda + col] : a[col * lda + row];
}
static void reference_syrk_s(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
int n, int k, float alpha, const float *a, int lda,
float beta, float *c, int ldc) {
for (int row = 0; row < n; ++row) {
for (int col = 0; col < n; ++col) {
if (!triangle_contains(uplo, row, col)) {
continue;
}
float sum = 0.0f;
for (int p = 0; p < k; ++p) {
sum += ref_load_f(order, trans, a, lda, row, p) * ref_load_f(order, trans, a, lda, col, p);
}
const int idx = index_of(order, ldc, row, col);
c[idx] = alpha * sum + beta * c[idx];
}
}
}
static void reference_syrk_d(CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
int n, int k, double alpha, const double *a, int lda,
double beta, double *c, int ldc) {
for (int row = 0; row < n; ++row) {
for (int col = 0; col < n; ++col) {
if (!triangle_contains(uplo, row, col)) {
continue;
}
double sum = 0.0;
for (int p = 0; p < k; ++p) {
sum += ref_load_d(order, trans, a, lda, row, p) * ref_load_d(order, trans, a, lda, col, p);
}
const int idx = index_of(order, ldc, row, col);
c[idx] = alpha * sum + beta * c[idx];
}
}
}
static void fill_float(float *dst, int size) {
for (int i = 0; i < size; ++i) {
dst[i] = (float)((i % 7) - 3) * 0.5f + (float)(i % 3);
}
}
static void fill_double(double *dst, int size) {
for (int i = 0; i < size; ++i) {
dst[i] = (double)((i % 11) - 5) * 0.25 + (double)(i % 5) * 0.1;
}
}
static void run_float_case(const char *name, CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
int n, int k, float alpha, float beta) {
++total;
const int lda = (trans == CblasNoTrans) ? k : n;
const int ldc = n;
float a[64];
float got[64];
float expected[64];
fill_float(a, lda * ((trans == CblasNoTrans) ? n : k));
fill_float(got, ldc * n);
memcpy(expected, got, sizeof(got));
reference_syrk_s(order, uplo, trans, n, k, alpha, a, lda, beta, expected, ldc);
syrk_s(order, uplo, trans, n, k, alpha, a, lda, beta, got, ldc);
for (int row = 0; row < n; ++row) {
for (int col = 0; col < n; ++col) {
const int idx = index_of(order, ldc, row, col);
if (triangle_contains(uplo, row, col)) {
if (!nearly_equal_f(got[idx], expected[idx])) {
++failures;
printf("[FAIL] %s -> mismatch at (%d,%d): got=%f expected=%f\n",
name, row, col, got[idx], expected[idx]);
return;
}
} else if (!nearly_equal_f(got[idx], expected[idx])) {
++failures;
printf("[FAIL] %s -> untouched triangle changed at (%d,%d)\n", name, row, col);
return;
}
}
}
printf("[PASS] %s\n", name);
}
static void run_double_case(const char *name, CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
int n, int k, double alpha, double beta) {
++total;
const int lda = (trans == CblasNoTrans) ? k : n;
const int ldc = n;
double a[64];
double got[64];
double expected[64];
fill_double(a, lda * ((trans == CblasNoTrans) ? n : k));
fill_double(got, ldc * n);
memcpy(expected, got, sizeof(got));
reference_syrk_d(order, uplo, trans, n, k, alpha, a, lda, beta, expected, ldc);
syrk_d(order, uplo, trans, n, k, alpha, a, lda, beta, got, ldc);
for (int row = 0; row < n; ++row) {
for (int col = 0; col < n; ++col) {
const int idx = index_of(order, ldc, row, col);
if (triangle_contains(uplo, row, col)) {
if (!nearly_equal_d(got[idx], expected[idx])) {
++failures;
printf("[FAIL] %s -> mismatch at (%d,%d): got=%lf expected=%lf\n",
name, row, col, got[idx], expected[idx]);
return;
}
} else if (!nearly_equal_d(got[idx], expected[idx])) {
++failures;
printf("[FAIL] %s -> untouched triangle changed at (%d,%d)\n", name, row, col);
return;
}
}
}
printf("[PASS] %s\n", name);
}
static void run_validation_checks(void) {
++total;
if (!syrk_validate(CblasColMajor, CblasUpper, CblasNoTrans, 8, 6, 6, 8) ||
syrk_validate(CblasColMajor, CblasUpper, CblasNoTrans, 8, 6, 5, 8) ||
syrk_validate(CblasRowMajor, CblasLower, CblasTrans, 8, 6, 7, 8)) {
++failures;
printf("[FAIL] validation rules\n");
return;
}
printf("[PASS] validation rules\n");
}
int main(void) {
puts("SYRK functional test suite");
run_validation_checks();
run_float_case("float col-major upper no-trans", CblasColMajor, CblasUpper, CblasNoTrans, 4, 3, 1.25f, 0.5f);
run_float_case("float col-major lower trans", CblasColMajor, CblasLower, CblasTrans, 4, 3, -0.75f, 1.0f);
run_float_case("float row-major upper no-trans", CblasRowMajor, CblasUpper, CblasNoTrans, 4, 3, 2.0f, -1.0f);
run_float_case("float row-major lower trans", CblasRowMajor, CblasLower, CblasTrans, 4, 3, 0.5f, 0.25f);
run_double_case("double col-major upper no-trans", CblasColMajor, CblasUpper, CblasNoTrans, 4, 3, 1.25, 0.5);
run_double_case("double col-major lower trans", CblasColMajor, CblasLower, CblasTrans, 4, 3, -0.75, 1.0);
run_double_case("double row-major upper no-trans", CblasRowMajor, CblasUpper, CblasNoTrans, 4, 3, 2.0, -1.0);
run_double_case("double row-major lower trans", CblasRowMajor, CblasLower, CblasTrans, 4, 3, 0.5, 0.25);
printf("\nTotal: %d, failed: %d\n", total, failures);
return failures == 0 ? 0 : 1;
}