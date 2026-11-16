// Deliberately non-synthesizable code, used to exercise lint_hls.
#include <cstdlib>
#include <vector>

int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

void vadd(float *a, float *b, float *c, int n) {
  float *tmp = (float *)malloc(n * sizeof(float));
  std::vector<float> v(n);
  for (int i = 0; i < n; i++) tmp[i] = a[i] + b[i];
  for (int i = 0; i < n; i++) c[i] = tmp[i];
  free(tmp);
}
