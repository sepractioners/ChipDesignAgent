#define M 64
#define N 64
#define K 64
#define TILE_SIZE 32

void gemm_hls(float A[M][K], float B[K][N], float C[M][N]) {
    // Local tile buffers to mimic systolic array registers
    float A_tile[TILE_SIZE][TILE_SIZE]; // M_TILE x K_TILE
    float B_tile[TILE_SIZE][TILE_SIZE]; // K_TILE x N_TILE
    float C_tile[TILE_SIZE][TILE_SIZE]; // M_TILE x N_TILE

    // Pragma: Partition all tile buffers for parallel access
    #pragma HLS ARRAY_PARTITION variable=A_tile complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A_tile complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_tile complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_tile complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=2

    // Loop for M tiles
    for (int m_t = 0; m_t < M / TILE_SIZE; m_t++) {
        for (int n_t = 0; n_t < N / TILE_SIZE; n_t++) {
            for (int k_t = 0; k_t < K / TILE_SIZE; k_t++) {
                
                // --- 1. Load Tiles ---
                
                // Load A_tile: A[m*T + m_local][k_local]
                for (int m_local = 0; m_local < TILE_SIZE; m_local++) {
                    for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                        A_tile[m_local][k_local] = A[(m_t * TILE_SIZE + m_local)][(k_t * TILE_SIZE + k_local)];
                    }
                }

                // Load B_tile: B[k_local][n_local]
                for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                    for (int n_local = 0; n_local < TILE_SIZE; n_local++) {
                        B_tile[k_local][n_local] = B[(k_t * TILE_SIZE + k_local)][(n_t * TILE_SIZE + n_local)];
                    }
                }

                // Initialize C_tile for accumulation
                for (int m_local = 0; m_local < TILE_SIZE; m_local++) {
                    for (int n_local = 0; n_local < TILE_SIZE; n_local++) {
                        C_tile[m_local][n_local] = 0.0f; 
                    }
                }
                
                // --- 2. Core Tile Computation (MAC loop) ---
                
                for (int m_local = 0; m_local < TILE_SIZE; m_local++) {
                    #pragma HLS UNROLL
                    for (int n_local = 0; n_local < TILE_SIZE; n_local++) {
                        #pragma HLS UNROLL
                        
                        // The innermost loop (k_local) performs the dot product accumulation
                        float sum = 0.0f;
                        
                        for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                            #pragma HLS PIPELINE II=1
                            sum += A_tile[m_local][k_local] * B_tile[k_local][n_local];
                        }
                        
                        C_tile[m_local][n_local] += sum;
                    }
                }
                
                // --- 3. Write Back Tile Result ---
                // Write C_tile back to the main output array C
                for (int m_local = 0; m_local < TILE_SIZE; m_local++) {
                    for (int n_local = 0; n_local < TILE_SIZE; n_local++) {
                        C[(m_t * TILE_SIZE + m_local)][(n_t * TILE_SIZE + n_local)] += C_tile[m_local][n_local];
                    }
                }
            }
        }
    }
}