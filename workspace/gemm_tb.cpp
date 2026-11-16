// Auto-generated equivalence testbench: gemm (reference) vs gemm_hls (HLS candidate)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>

static std::mt19937 tb_rng(42);
static float tb_rnd_real() { return std::uniform_real_distribution<float>(-1.0f, 1.0f)(tb_rng); }

template <typename TbElem>
static int tb_compare(const char* tb_name, const TbElem* tb_ref, const TbElem* tb_cand, size_t tb_n, bool tb_is_float) {
  int tb_errs = 0;
  double tb_max = 0.0;
  for (size_t tb_i = 0; tb_i < tb_n; ++tb_i) {
    double tb_r = (double)tb_ref[tb_i], tb_c = (double)tb_cand[tb_i];
    double tb_err = std::fabs(tb_r - tb_c);
    double tb_tol = tb_is_float ? 0.0001 * (1.0 + std::fabs(tb_r)) : 0.0;
    if (tb_err > tb_max) tb_max = tb_err;
    if (!(tb_err <= tb_tol)) {
      if (tb_errs < 5) std::printf("MISMATCH %s[%zu]: ref=%.8g cand=%.8g\n", tb_name, tb_i, tb_r, tb_c);
      ++tb_errs;
    }
  }
  std::printf("%s: %zu elements, %d mismatches, max abs error %.3g\n", tb_name, tb_n, tb_errs, tb_max);
  return tb_errs;
}

#include "gemm.c"
#include "gemm_hls.cpp"

static float ref_A[M][K];
static float cand_A[M][K];
static float ref_B[K][N];
static float cand_B[K][N];
static float ref_C[M][N];
static float cand_C[M][N];

int main() {
  { float* tb_p = (float*)ref_A; for (size_t tb_i = 0; tb_i < sizeof(ref_A)/sizeof(float); ++tb_i) tb_p[tb_i] = tb_rnd_real(); std::memcpy(cand_A, ref_A, sizeof(ref_A)); }
  { float* tb_p = (float*)ref_B; for (size_t tb_i = 0; tb_i < sizeof(ref_B)/sizeof(float); ++tb_i) tb_p[tb_i] = tb_rnd_real(); std::memcpy(cand_B, ref_B, sizeof(ref_B)); }
  { float* tb_p = (float*)ref_C; for (size_t tb_i = 0; tb_i < sizeof(ref_C)/sizeof(float); ++tb_i) tb_p[tb_i] = tb_rnd_real(); std::memcpy(cand_C, ref_C, sizeof(ref_C)); }
  gemm(ref_A, ref_B, ref_C);
  gemm_hls(cand_A, cand_B, cand_C);
  int tb_errors = 0;
  tb_errors += tb_compare("C", (const float*)ref_C, (const float*)cand_C, sizeof(ref_C)/sizeof(float), true);
  if (tb_errors) { std::printf("FAIL: %d mismatches\n", tb_errors); return 1; }
  std::printf("PASS\n");
  return 0;
}
