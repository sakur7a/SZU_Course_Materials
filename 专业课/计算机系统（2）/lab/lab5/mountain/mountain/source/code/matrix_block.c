#include <sys/time.h> 
#include <unistd.h> 
#include <stdlib.h>
#include <stdio.h> 

#define BLOCK_SIZE 32 // 定义块大小

int main(int argc, char *argv[]) 
{
    float *a, *b, *c;
    long int i, j, k, size, m;
    struct timeval time1, time2; 
    
    if (argc < 2) { 
        printf("\n\tUsage:%s <Row of square matrix>\n", argv[0]); 
        exit(-1); 
    }

    size = atoi(argv[1]);
    m = size * size;
    a = (float*)malloc(sizeof(float) * m); 
    b = (float*)malloc(sizeof(float) * m); 
    c = (float*)malloc(sizeof(float) * m); 

    for (i = 0; i < size; i++) { 
        for (j = 0; j < size; j++) { 
            a[i*size+j] = (float)(rand() % 1000 / 100.0); 
            b[i*size+j] = (float)(rand() % 1000 / 100.0); 
            c[i*size+j] = 0.0f; // 在这里顺便把c初始化清零
        }
    }
    
    gettimeofday(&time1, NULL);
    
    // 核心：分块矩阵乘法（Blocking）结合 i-k-j 顺序
    long int i1, j1, k1;
    for (i = 0; i < size; i += BLOCK_SIZE) {
        for (k = 0; k < size; k += BLOCK_SIZE) {
            for (j = 0; j < size; j += BLOCK_SIZE) {
                // 处理一个小块
                for (i1 = i; i1 < i + BLOCK_SIZE && i1 < size; i1++) {
                    for (k1 = k; k1 < k + BLOCK_SIZE && k1 < size; k1++) {
                        float temp_a = a[i1*size + k1];
                        for (j1 = j; j1 < j + BLOCK_SIZE && j1 < size; j1++) {
                            c[i1*size + j1] += temp_a * b[k1*size + j1];
                        }
                    }
                }
            }
        }
    }
    
    gettimeofday(&time2, NULL);    
    
    time2.tv_sec -= time1.tv_sec;
    time2.tv_usec -= time1.tv_usec; 
    if (time2.tv_usec < 0L) { 
        time2.tv_usec += 1000000L; 
        time2.tv_sec -= 1; 
    } 
   
    double time_ms = time2.tv_sec * 1000.0 + time2.tv_usec / 1000.0;
    printf("N = %ld, Block Time = %.3f ms\n", size, time_ms); 
    
    free(a);
    free(b);
    free(c);
    
    return 0; 
}