// Hand-written HLS version of examples/gemm.c (i-k-j order, local C row buffer).
#define M 64
#define N 64
#define K 64

void gemm_hls(float A[M][K], float B[K][N], float C[M][N]) {
#pragma HLS INTERFACE mode=m_axi port=A bundle=gmem0 depth=4096
#pragma HLS INTERFACE mode=m_axi port=B bundle=gmem1 depth=4096
#pragma HLS INTERFACE mode=m_axi port=C bundle=gmem0 depth=4096
#pragma HLS INTERFACE mode=s_axilite port=return
  float row[N];
#pragma HLS ARRAY_PARTITION variable=row type=cyclic factor=16
row_loop:
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) row[j] = 0;
    for (int k = 0; k < K; k++) {
      float a = A[i][k];
      for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
        row[j] += a * B[k][j];
      }
    }
    for (int j = 0; j < N; j++) C[i][j] = row[j];
  }
}
