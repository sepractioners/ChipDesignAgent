// Dense matrix multiply C = A * B (PolyBench-style). Linear-algebra pattern.
#define M 64
#define N 64
#define K 64

void gemm(float A[M][K], float B[K][N], float C[M][N]) {
#pragma scop
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
      for (int k = 0; k < K; k++)
        C[i][j] += A[i][k] * B[k][j];
    }
#pragma endscop
}
