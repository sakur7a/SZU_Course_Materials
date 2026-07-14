/*
 * tlb_measure.c - Measure TLB size by varying number of pages accessed
 *
 * Strategy: Access one element per page with a fixed stride of one page (4KB).
 * As the number of pages exceeds TLB capacity, latency increases sharply.
 * The inflection point reveals the TLB size.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>

static int gettimeofday(struct timeval *tv, void *tz)
{
    FILETIME ft;
    unsigned long long ticks;
    (void)tz;
    GetSystemTimeAsFileTime(&ft);
    ticks = ((unsigned long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    ticks -= 116444736000000000ULL;
    tv->tv_sec = (long)(ticks / 10000000ULL);
    tv->tv_usec = (long)((ticks % 10000000ULL) / 10);
    return 0;
}
#else
#include <sys/time.h>
#endif

static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtscp() {
    unsigned int lo, hi;
    __asm__ volatile ("rdtscp" : "=a"(lo), "=d"(hi) : : "%ecx");
    return ((uint64_t)hi << 32) | lo;
}

#define PAGE_SIZE 4096
#define ELEMS_PER_PAGE (PAGE_SIZE / sizeof(double))  /* 512 elements per page */

/*
 * Measure average latency (cycles) when accessing 'num_pages' different pages,
 * one element per page, with sequential page access pattern.
 */
double measure_tlb_latency(volatile double *data, int num_pages, int repeats) {
    uint64_t total_cycles = 0;

    /* Warm up - touch all pages once */
    for (int p = 0; p < num_pages; p++)
        (void)data[p * ELEMS_PER_PAGE];

    for (int r = 0; r < repeats; r++) {
        uint64_t start = rdtsc();
        for (int p = 0; p < num_pages; p++)
            (void)data[p * ELEMS_PER_PAGE];
        uint64_t end = rdtscp();
        total_cycles += (end - start);
    }

    return (double)total_cycles / (repeats * num_pages);
}

/*
 * Measure latency with random access pattern across pages
 * (to defeat hardware prefetching across pages)
 */
double measure_tlb_latency_random(volatile double *data, int num_pages, int *order, int repeats) {
    uint64_t total_cycles = 0;

    /* Warm up */
    for (int p = 0; p < num_pages; p++)
        (void)data[order[p] * ELEMS_PER_PAGE];

    for (int r = 0; r < repeats; r++) {
        uint64_t start = rdtsc();
        for (int p = 0; p < num_pages; p++)
            (void)data[order[p] * ELEMS_PER_PAGE];
        uint64_t end = rdtscp();
        total_cycles += (end - start);
    }

    return (double)total_cycles / (repeats * num_pages);
}

/* Measure CPU frequency */
double measure_mhz() {
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    uint64_t c1 = rdtsc();
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000000L;
    nanosleep(&ts, NULL);
    uint64_t c2 = rdtsc();
    gettimeofday(&t2, NULL);
    double elapsed = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
    return (c2 - c1) / elapsed / 1e6;
}

/* Fisher-Yates shuffle */
void shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

int main() {
    double mhz = measure_mhz();
    fprintf(stderr, "CPU frequency: %.1f MHz\n", mhz);

    /* Allocate enough memory for many pages */
    int max_pages = 8192; /* 8192 pages = 32 MB */
    int max_elems = max_pages * ELEMS_PER_PAGE;
    volatile double *data = (volatile double *)malloc(max_elems * sizeof(double));
    if (!data) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    /* Initialize */
    for (int i = 0; i < max_elems; i++)
        data[i] = (double)i;

    /* Prepare random access order */
    int *order = (int *)malloc(max_pages * sizeof(int));
    for (int i = 0; i < max_pages; i++)
        order[i] = i;

    fprintf(stderr, "Testing TLB with up to %d pages (%d MB)\n\n", max_pages, max_pages * 4 / 1024);

    /* Print header */
    printf("num_pages,size_KB,sequential_cycles,random_cycles\n");

    /* Test various numbers of pages */
    int page_counts[] = {1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128,
                         160, 192, 224, 256, 320, 384, 448, 512,
                         640, 768, 896, 1024, 1280, 1536, 1792, 2048,
                         2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192};
    int num_tests = sizeof(page_counts) / sizeof(page_counts[0]);

    for (int t = 0; t < num_tests; t++) {
        int pages = page_counts[t];
        if (pages > max_pages) break;

        /* Shuffle order for random access */
        srand(42);
        shuffle(order, pages);

        double seq_lat = measure_tlb_latency(data, pages, 5);
        double rand_lat = measure_tlb_latency_random(data, pages, order, 5);

        printf("%d,%.1f,%.1f,%.1f\n", pages, pages * 4.0, seq_lat, rand_lat);
        fflush(stdout);
    }

    free((void*)data);
    free(order);
    fprintf(stderr, "\nTLB measurement complete.\n");
    return 0;
}
