#include <stdio.h>
#include <stdlib.h>
//#include <thread>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 2048
#define K 2048
#define N 2048


void initMatrix(float *A, int m, int n) {
    for (int i = 0; i < m * n; ++i) {
        A[i] = i;
    }
    return;
}

void gemm(float *C, float *A, float *B, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i + j * m] = 0.0f;
            for (int l = 0; l < k; ++l) {
                C[i + j * m] += A[i + l * m] * B[l + j * k];
            }
        }
    }
}

void gemmcublas(float *C, float *A, float *B, int m, int k, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    float *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr};
    cudaMalloc(&d_A, sizeof(float) * m * k);
    cudaMalloc(&d_B, sizeof(float) * k * n);
    cudaMalloc(&d_C, sizeof(float) * m * n);

    cublasSetMatrix(m, n, sizeof(float), A, m, d_A, m);
    cublasSetMatrix(m, n, sizeof(float), B, k, d_B, k);

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    cublasGetMatrix(m, n, sizeof(float), d_C, m, C, m);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}

void getSubmatrixPointer(float *A, int mDiv2, int nDiv2, int ld, float *&A_11, float *&A_12, float *&A_21, float *&A_22) {
    A_11 = A;
    A_21 = A + mDiv2;
    A_12 = A + nDiv2 * ld;
    A_22 = A_12 + mDiv2;
}

void matrixAdd(float *C, float *A, float *B, int m, int n, int ldC, int ldA, int ldB) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i + j * ldC] = A[i + j * ldA] + B[i + j * ldB];
        }
    }
}

void matrixMinus(float *C, float *A, float *B, int m, int n, int ldC, int ldA, int ldB) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i + j * ldC] = A[i + j * ldA] - B[i + j * ldB];
        }
    }
}

void matrixCopy(float *C, float *A, int m, int n, int ldC, int ldA) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i + j * ldC] = A[i + j * ldA];
        }
    }
}

void gemmstrassen(float *C, float *A, float *B, int m, int k, int n) {
    int mDiv2 = m / 2;
    int kDiv2 = k / 2;
    int nDiv2 = n / 2;

    float *A_11, *A_12, *A_21, *A_22;
    getSubmatrixPointer(A, mDiv2, kDiv2, m, A_11, A_12, A_21, A_22);

    float *B_11, *B_12, *B_21, *B_22;
    getSubmatrixPointer(B, kDiv2, nDiv2, k, B_11, B_12, B_21, B_22);

    float *M_1A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_1B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M1 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_2A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_2B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M2 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_3A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_3B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M3 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_4A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_4B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M4 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_5A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_5B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M5 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_6A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_6B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M6 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_7A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_7B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M7 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
#pragma omp parallel
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2);
            }

#pragma omp section
            {
                matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, k);
                gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2);
            }
#pragma omp section
            {
                matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, m);
                matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2);
            }
#pragma omp section
            {
                matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, m);
                matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2);
            }
#pragma omp section
            {
                matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, m, m);
                matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, k);
                gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2);
            }
#pragma omp section
            {
                matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2);
            }
#pragma omp section
            {
                matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2);
            }
        }
    }
    //float *M_1A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    //matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, m, m);
    //float *M_1B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    //matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, k, k);
    //float *M1 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    //gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2);

    //float *M_2A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    //matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, m, m);
    //float *M_2B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    //matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, k);
    //float *M2 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    //gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2);

    //float *M_3A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    //matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, m);
    //float *M_3B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    //matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, k, k);
    //float *M3 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    //gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2);

    //float *M_4A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    //matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, m);
    //float *M_4B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    //matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, k, k);
    //float *M4 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    //gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2);

    //float *M_5A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    //matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, m, m);
    //float *M_5B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    //matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, k);
    //float *M5 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    //gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2);

    //float *M_6A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    //matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, m, m);
    //float *M_6B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    //matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, k, k);
    //float *M6 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    //gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2);

    //float *M_7A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    //matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, m, m);
    //float *M_7B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    //matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, k, k);
    //float *M7 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    //gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2);

    float *C_11, *C_12, *C_21, *C_22;
    getSubmatrixPointer(C, mDiv2, nDiv2, m, C_11, C_12, C_21, C_22);

#pragma omp parallel
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(C_11, M1, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);
                matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, m, m, mDiv2);
                matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, m, m, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_21, M2, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_12, M3, M5, mDiv2, nDiv2, m, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixMinus(C_22, M1, M2, mDiv2, nDiv2, m, mDiv2, mDiv2);
                matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, m, m, mDiv2);
                matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, m, m, mDiv2);
            }
        }
    }

    //matrixAdd(C_11, M1, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);
    //matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, m, m, mDiv2);
    //matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, m, m, mDiv2);

    //matrixAdd(C_21, M2, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);

    //matrixAdd(C_12, M3, M5, mDiv2, nDiv2, m, mDiv2, mDiv2);

    //matrixMinus(C_22, M1, M2, mDiv2, nDiv2, m, mDiv2, mDiv2);
    //matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, m, m, mDiv2);
    //matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, m, m, mDiv2);

    free(M_1A);
    free(M_1B);
    free(M1);
    free(M_2A);
    free(M_2B);
    free(M2);
    free(M_3A);
    free(M_3B);
    free(M3);
    free(M_4A);
    free(M_4B);
    free(M4);
    free(M_5A);
    free(M_5B);
    free(M5);
    free(M_6A);
    free(M_6B);
    free(M6);
    free(M_7A);
    free(M_7B);
    free(M7);
}

//void parallelStrassen(float *C, float *A, float *B, int m, int k, int n, int blockSizeM, int blockSizeK, int blockSizeN) {
//    int blockNumM = m / blockSizeM;
//    int blockNumK = k / blockSizeK;
//    int blockNumN = n / blockSizeN;
//
//    std::thread *threadPointer = (std::thread*)malloc(sizeof(std::thread) * blockNumM * blockNumN);
//    for (int i = 0; i < blockNumM; ++i) {
//        for (int j = 0; j < blockNumN; ++j) {
//            threadPointer[i * blockNumN + j] = new std::thread(gemmstrassen, )
//        }
//    }
//
//}

void test(float *A, float *B, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float diff = A[i * m + j] - B[i * m + j];
            diff /= A[i * m + j];
            if (diff > 0.001 || diff < -0.001) {
                printf("wrong result. i:%d j:%d\n", i, j);
            }
        }
    }
}

int main() {
    float *h_A = (float*)malloc(sizeof(float) * M * K);
    float *h_B = (float*)malloc(sizeof(float) * K * N);
    float *h_C = (float*)malloc(sizeof(float) * M * N);
    float *h_CTest = (float*)malloc(sizeof(float) * M * N);

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gemmcublas(h_C, h_A, h_B, M, K, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("gemmcublas times %f\n", time);

    gemm(h_CTest, h_A, h_B, M, K, N);

    cudaEventRecord(start, 0);
    gemmstrassen(h_C, h_A, h_B, M, K, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("gemmstarssen times %f\n", time);
    test(h_C, h_CTest, M, N);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CTest);

    return 0;
}