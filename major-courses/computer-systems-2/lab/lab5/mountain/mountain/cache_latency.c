/*
 * cache_latency.c - Measure cache hierarchy on x86.
 *
 * The program uses a dependent pointer-chasing access pattern.  Each load
 * produces the index of the next load, so the CPU cannot hide the latency with
 * ordinary out-of-order execution or simple hardware prefetching.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MIN_BYTES (1 << 10)
#define MAX_BYTES (1 << 26)
#define MIN_STRIDE_ELEMS 1
#define MAX_STRIDE_ELEMS 64
#define STEPS_PER_TEST 1000000

static inline uint64_t rdtsc_start(void)
{
    unsigned int lo, hi;
    __asm__ volatile("cpuid\n\t"
                     "rdtsc\n\t"
                     : "=a"(lo), "=d"(hi)
                     :
                     : "%rbx", "%rcx", "memory");
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_stop(void)
{
    unsigned int lo, hi;
    __asm__ volatile("rdtscp\n\t"
                     "mov %%eax, %0\n\t"
                     "mov %%edx, %1\n\t"
                     "cpuid\n\t"
                     : "=r"(lo), "=r"(hi)
                     :
                     : "%rax", "%rbx", "%rcx", "%rdx", "memory");
    return ((uint64_t)hi << 32) | lo;
}

static double measure_mhz(void)
{
    struct timeval t1, t2;
    struct timespec ts;

    ts.tv_sec = 0;
    ts.tv_nsec = 100000000L;

    gettimeofday(&t1, NULL);
    uint64_t c1 = rdtsc_start();
    nanosleep(&ts, NULL);
    uint64_t c2 = rdtsc_stop();
    gettimeofday(&t2, NULL);

    double elapsed = (t2.tv_sec - t1.tv_sec) +
                     (t2.tv_usec - t1.tv_usec) / 1000000.0;
    return (c2 - c1) / elapsed / 1000000.0;
}

/*
 * Build one cyclic linked list inside "next".
 * The participating elements are separated by stride_elems.  Their order is
 * shuffled, so a stride of 64 cache lines is not seen by the prefetcher as a
 * simple ascending stream.
 */
static int build_chase_list(int *next, int elems, int stride_elems)
{
    int count = elems / stride_elems;
    if (count < 1)
        count = 1;

    int *indices = (int *)malloc((size_t)count * sizeof(int));
    if (!indices) {
        fprintf(stderr, "failed to allocate index list\n");
        exit(1);
    }

    for (int i = 0; i < count; i++)
        indices[i] = i * stride_elems;

    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    for (int i = 0; i < count; i++)
        next[indices[i]] = indices[(i + 1) % count];

    int start = indices[0];
    free(indices);
    return start;
}

static double measure_latency(int *next, int elems, int stride_elems)
{
    srand(2026 + elems * 17 + stride_elems);
    int index = build_chase_list(next, elems, stride_elems);

    for (int i = 0; i < elems; i += stride_elems)
        index = next[index];

    int steps = STEPS_PER_TEST;
    int active = elems / stride_elems;
    if (active < 1)
        active = 1;
    if (active < steps / 4)
        steps = active * 64;
    if (steps < 10000)
        steps = 10000;

    uint64_t start = rdtsc_start();
    for (int i = 0; i < steps; i++)
        index = next[index];
    uint64_t end = rdtsc_stop();

    volatile int sink = index;
    (void)sink;
    return (double)(end - start) / steps;
}

static double measure_throughput(double *data, int elems, int stride_elems,
                                 double mhz)
{
    volatile double sink = 0.0;
    int accesses = elems / stride_elems;
    if (accesses < 1)
        accesses = 1;

    for (int i = 0; i < elems; i += stride_elems)
        sink += data[i];

    uint64_t start = rdtsc_start();
    for (int r = 0; r < 5; r++) {
        double sum = 0.0;
        for (int i = 0; i < elems; i += stride_elems)
            sum += data[i];
        sink = sum;
    }
    uint64_t end = rdtsc_stop();

    double cycles = (double)(end - start) / 5.0;
    double seconds = cycles / (mhz * 1000000.0);
    double bytes_read = (double)accesses * sizeof(double);
    (void)sink;
    return bytes_read / seconds / 1000000.0;
}

int main(void)
{
    double mhz = measure_mhz();
    int max_int_elems = MAX_BYTES / (int)sizeof(int);
    int max_double_elems = MAX_BYTES / (int)sizeof(double);

    int *next = (int *)malloc((size_t)max_int_elems * sizeof(int));
    double *data = (double *)malloc((size_t)max_double_elems * sizeof(double));
    if (!next || !data) {
        fprintf(stderr, "failed to allocate arrays\n");
        free(next);
        free(data);
        return 1;
    }

    for (int i = 0; i < max_double_elems; i++)
        data[i] = (double)i;

    fprintf(stderr, "Measured CPU frequency: %.1f MHz\n", mhz);
    fprintf(stderr, "Array allocated: %d MB\n", MAX_BYTES / (1024 * 1024));

    printf("Memory Mountain (MB/sec)\n");
    printf("CPU_freq_MHz\t%.1f\n", mhz);
    printf("\t");
    for (int stride = MIN_STRIDE_ELEMS; stride <= MAX_STRIDE_ELEMS; stride++)
        printf("s%d\t", stride);
    printf("\n");

    for (int size = MAX_BYTES; size >= MIN_BYTES; size >>= 1) {
        int elems = size / (int)sizeof(double);
        if (size >= (1 << 20))
            printf("%dm\t", size / (1 << 20));
        else
            printf("%dk\t", size / 1024);

        for (int stride = MIN_STRIDE_ELEMS; stride <= MAX_STRIDE_ELEMS; stride++)
            printf("%.1f\t", measure_throughput(data, elems, stride, mhz));
        printf("\n");
    }

    fprintf(stderr, "\n=== Cache Latency Data (cycles per access) ===\n");
    fprintf(stderr, "Size_KB\tStride1\tStride2\tStride4\tStride8\tStride16\tStride32\tStride64\n");

    int sizes_kb[] = {
        1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
        768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
        24576, 32768, 49152, 65536
    };
    int strides[] = {1, 2, 4, 8, 16, 32, 64};
    int size_count = (int)(sizeof(sizes_kb) / sizeof(sizes_kb[0]));
    int stride_count = (int)(sizeof(strides) / sizeof(strides[0]));

    for (int si = 0; si < size_count; si++) {
        int bytes = sizes_kb[si] * 1024;
        int elems = bytes / (int)sizeof(int);
        fprintf(stderr, "%.1f\t", (double)sizes_kb[si]);
        for (int sj = 0; sj < stride_count; sj++) {
            double latency = measure_latency(next, elems, strides[sj]);
            fprintf(stderr, "%.1f\t", latency);
        }
        fprintf(stderr, "\n");
    }

    free(next);
    free(data);
    fprintf(stderr, "\nDone.\n");
    return 0;
}
