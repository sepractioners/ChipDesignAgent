// One sweep of a 5-point 2-D Jacobi stencil. Stencil pattern.
#define H 128
#define W 128

void jacobi2d(float A[H][W], float B[H][W]) {
  for (int i = 1; i < H - 1; i++)
    for (int j = 1; j < W - 1; j++)
      B[i][j] = 0.2f * (A[i][j] + A[i][j - 1] + A[i][j + 1] + A[i + 1][j] + A[i - 1][j]);
}
