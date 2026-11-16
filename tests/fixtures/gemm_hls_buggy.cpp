// Same as gemm_hls.cpp but with an off-by-one in the k loop (skips k = K-1).
#define M 64
#define N 64
#define K 64

void gemm(float A[M][K], float B[K][N], float C[M][N]) {
  float row[N];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) row[j] = 0;
    for (int k = 0; k < K - 1; k++)
      for (int j = 0; j < N; j++) row[j] += A[i][k] * B[k][j];
    for (int j = 0; j < N; j++) C[i][j] = row[j];
  }
}
