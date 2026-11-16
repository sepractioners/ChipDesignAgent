#include <cmath>

// Define constants from the original kernel
#define M 64
#define N 64
#define K 64

// Define the tiling factor T (32)
#define T 32

/**
 * @brief Optimized GEMM implementation using tiling and deep pipelining.
 * 
 * Implements C = A * B + C using 32x32x32 tiling and local buffers.
 * 
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output/Accumulation matrix C (M x N)
 */
void gemm_hls(float A[M][K], float B[K][N], float C[M][N]) {
    // Local cache for the C tile (T x T). 
    // This significantly reduces off-chip memory access within the inner loops.
    float C_local[T][T]; 

    // --- Tiled Loops ---

    // Outer loop over M dimension (Row blocks)
    // We don't need explicit pragmas here as the loops cover the whole range.
    for (int m_start = 0; m_start < M; m_start += T) {
        // Outer loop over N dimension (Column blocks)
        for (int n_start = 0; n_start < N; n_start += T) {
            // Outer loop over K dimension (Inner blocks)
            // This loop accumulates the partial result.
            for (int k_start = 0; k_start < K; k_start += T) {
                
                // --- Initialization/Accumulation Start ---
                
                // 1. Load the initial C tile block into the local memory.
                for (int i = 0; i < T; i++) {
                    for (int j = 0; j < T; j++) {
                        // C_local accumulates: C_local = C_local + A_tile * B_tile
                        C_local[i][j] = C[m_start + i][n_start + j];
                    }
                }
                
                // 2. Inner tile loops: i (M), j (N)
                // We iterate over the tile dimensions
                // UNROLL i and j fully to expose loop-level parallelism for the MAC operation.
                #pragma HLS UNROLL
                for (int i = 0; i < T; i++) {
                    #pragma HLS UNROLL
                    for (int j = 0; j < T; j++) {
                        
                        // 3. The innermost MAC loop: k (K dimension)
                        // Target PIPELINE on the K axis to maximize throughput.
                        #pragma HLS PIPELINE II=1
                        for (int k = 0; k < T; k++) {
                            
                            // MAC Operation: C_local[i][j] += A[i][k] * B[k][j]
                            // Note: Indexing here is local to the current tile (relative to A[m_start+i], B[k_start+k], etc.)
                            
                            float a_val = A[m_start + i][k_start + k];
                            float b_val = B[k_start + k][n_start + j];
                            
                            C_local[i][j] += a_val * b_val;
                        } // end k
                    } // end j
                } // end i
                
                // --- Write back the result (T x T block) ---
                // Write back the final accumulated tile result C_local to the global memory C
                for (int i = 0; i < T; i++) {
                    for (int j = 0; j < T; j++) {
                        C[m_start + i][n_start + j] = C_local[i][j];
                    }
                }
            } // end k_start
        } // end n_start
    } // end m_start
}