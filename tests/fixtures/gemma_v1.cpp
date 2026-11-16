// First HLS version Gemma 4 wrote for examples/gemm.c (kept verbatim as a lint fixture).
#include <ap_int.h>

#define M 64
#define N 64
#define K 64

void gemm(float A[M][K], float B[K][N], float C[M][N]) {
#pragma HLS INTERFACE m_axi port=A bundle=peripheral offset=slave depth=1024
#pragma HLS INTERFACE m_axi port=B bundle=peripheral offset=slave depth=1024
#pragma HLS INTERFACE m_axi port=C bundle=peripheral offset=slave depth=1024
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS PIPELINE II=1

  for (int i = 0; i < M; i++) {
    #pragma HLS ARRAY_PARTITION variable=C complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C complete dim=2
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}
