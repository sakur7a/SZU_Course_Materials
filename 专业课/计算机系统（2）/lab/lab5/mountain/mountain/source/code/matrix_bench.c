/*
 * matrix_bench.c - Comprehensive matrix multiplication benchmark
 * Compares: Normal (i-j-k), Optimized (i-k-j), Blocking (tiled)
 * Runs multiple sizes and outputs CSV for analysis
 */
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BLOCK_SIZE 32
#define NUM_TRIALS 1

static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Normal i-j-k order */
void matmul_normal(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += a[i*n+k] * b[k*n+j];
            c[i*n+j] = sum;
        }
    }
}

/* Optimized i-k-j order (better spatial locality for B) */
void matmul_ikj(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            c[i*n+j] = 0.0f;

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float temp_a = a[i*n+k];
            for (int j = 0; j < n; j++)
                c[i*n+j] += temp_a * b[k*n+j];
        }
    }
}

/* Blocking (tiling) + i-k-j */
void matmul_block(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n*n; i++)
        c[i] = 0.0f;

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                int i_max = i + BLOCK_SIZE < n ? i + BLOCK_SIZE : n;
                int k_max = k + BLOCK_SIZE < n ? k + BLOCK_SIZE : n;
                int j_max = j + BLOCK_SIZE < n ? j + BLOCK_SIZE : n;
                for (int i1 = i; i1 < i_max; i1++) {
                    for (int k1 = k; k1 < k_max; k1++) {
                        float temp_a = a[i1*n + k1];
                        for (int j1 = j; j1 < j_max; j1++)
                            c[i1*n + j1] += temp_a * b[k1*n + j1];
                    }
                }
            }
        }
    }
}

int main() {
    int sizes[] = {100, 500, 1000, 1500, 2000, 2500, 3000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("N,Normal_ms,IKJ_ms,Block_ms\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        long m = (long)n * n;

        float *a = (float*)malloc(sizeof(float) * m);
        float *b = (float*)malloc(sizeof(float) * m);
        float *c = (float*)malloc(sizeof(float) * m);

        srand(42);
        for (long i = 0; i < m; i++) {
            a[i] = (float)(rand() % 1000) / 100.0f;
            b[i] = (float)(rand() % 1000) / 100.0f;
        }

        /* Normal */
        double t_normal = 1e18;
        for (int t = 0; t < NUM_TRIALS; t++) {
            double start = get_time_ms();
            matmul_normal(a, b, c, n);
            double elapsed = get_time_ms() - start;
            if (elapsed < t_normal) t_normal = elapsed;
        }

        /* IKJ */
        double t_ikj = 1e18;
        for (int t = 0; t < NUM_TRIALS; t++) {
            double start = get_time_ms();
            matmul_ikj(a, b, c, n);
            double elapsed = get_time_ms() - start;
            if (elapsed < t_ikj) t_ikj = elapsed;
        }

        /* Block */
        double t_block = 1e18;
        for (int t = 0; t < NUM_TRIALS; t++) {
            double start = get_time_ms();
            matmul_block(a, b, c, n);
            double elapsed = get_time_ms() - start;
            if (elapsed < t_block) t_block = elapsed;
        }

        printf("%d,%.3f,%.3f,%.3f\n", n, t_normal, t_ikj, t_block);
        fflush(stdout);

        free(a); free(b); free(c);
    }

    return 0;
}
