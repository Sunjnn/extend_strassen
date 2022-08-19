#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <omp.h>
#include <math.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 2048
#define K 2048
#define N 2048


#define CHECKCUDA(call)                                                     \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:$d", __FILE__, __LINE__);                         \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));   \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

#define CHECKCUBLAS(call) {                                                                                             \
    const cublasStatus_t error = call;                                                                                  \
    if (error != CUBLAS_STATUS_SUCCESS) {                                                                               \
        printf("Error: %s:%d", __FILE__, __LINE__);                                                                     \
        printf("code: %d, name: %s, string: %s\n", error, cublasGetStatusName(error), cublasGetStatusString(error));    \
    }                                                                                                                   \
}                                                                                                                       \

#define EXIT() {                    \
    printf("Enter to exit:");      \
    getchar();                      \
}                                   \

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

void gemmcublas(float *C, float *A, float *B, int m, int k, int n, cublasHandle_t handle, cudaStream_t stream) {
    cublasStatus_t cublasStatus = CUBLAS_STATUS_SUCCESS;
    cublasStatus = cublasSetStream(handle, stream);
    CHECKCUBLAS(cublasStatus);

    //void *workspace{ nullptr };
    //CHECKCUDA(cudaMallocAsync(&workspace, 1024 * 1024, stream));
    //CHECKCUBLAS(cublasSetWorkspace(handle, workspace, 1024 * 1024));

    cudaError_t cudaStatus;
    float *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr};
    cudaStatus = cudaMallocAsync(&d_A, sizeof(float) * m * k, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaMallocAsync(&d_B, sizeof(float) * k * n, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaMallocAsync(&d_C, sizeof(float) * m * n, stream);
    CHECKCUDA(cudaStatus);

    cublasStatus = cublasSetMatrixAsync(m, k, sizeof(float), A, m, d_A, m, stream);
    CHECKCUBLAS(cublasStatus);
    cublasStatus = cublasSetMatrixAsync(k, n, sizeof(float), B, k, d_B, k, stream);
    CHECKCUBLAS(cublasStatus);

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    cublasStatus = cublasGetMatrixAsync(m, n, sizeof(float), d_C, m, C, m, stream);
    CHECKCUBLAS(cublasStatus);

    cudaStatus = cudaFreeAsync(d_A, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaFreeAsync(d_B, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaFreeAsync(d_C, stream);
    CHECKCUDA(cudaStatus);
}

void gemmcublas(float *C, float *A, float *B, int m, int k, int n, cublasHandle_t handle) {
    float *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr};
    cudaMalloc(&d_A, sizeof(float) * m * k);
    cudaMalloc(&d_B, sizeof(float) * k * n);
    cudaMalloc(&d_C, sizeof(float) * m * n);

    cublasSetMatrix(m, k, sizeof(float), A, m, d_A, m);
    cublasSetMatrix(k, n, sizeof(float), B, k, d_B, k);

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    cublasGetMatrix(m, n, sizeof(float), d_C, m, C, m);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void getSubmatrixPointer(float *A, int mDiv2, int nDiv2, int ld, float *&A_11, float *&A_12, float *&A_21, float *&A_22) {
    A_11 = A;
    A_21 = A + mDiv2;
    A_12 = A + nDiv2 * ld;
    A_22 = A_12 + mDiv2;
}

class blockMatrix {
public:
    float **pointers = nullptr;
    int dimM = 0;
    int dimN = 0;
    blockMatrix();
    blockMatrix(float *A, int m, int n, int ldA, int blockM, int blockN);
    float *getBlockMatrix(int blockI, int blockJ);
};

blockMatrix::blockMatrix() {}

blockMatrix::blockMatrix(float *A, int m, int n, int ldA, int blockM, int blockN) {
    dimM = ceil(m / blockM);
    dimN = ceil(n / blockN);

    pointers = (float**)malloc(sizeof(float*) * dimM * dimN);
    for (int i = 0; i < dimM; ++i) {
        for (int j = 0; j < dimN; ++j) {
            pointers[i + j * dimM] = A + i * blockM + j * blockN * ldA;
        }
    }
}

float *blockMatrix::getBlockMatrix(int blockI, int blockJ) {
    return pointers[blockI + blockJ * dimM];
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

    cublasHandle_t *handleArray = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cublasCreate(handleArray + i);
    }

    cudaStream_t *streamArray = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cudaStreamCreate(streamArray + i);
    }

    int id;
#pragma omp parallel private(id)
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0], streamArray[0]);
                //gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0]);
            }

#pragma omp section
            {
                matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, k);
                gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1], streamArray[1]);
                //gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1]);
            }
#pragma omp section
            {
                matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, m);
                matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2], streamArray[2]);
                //gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2]);
            }
#pragma omp section
            {
                matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, m);
                matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3], streamArray[3]);
                //gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3]);
            }
#pragma omp section
            {
                matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, m, m);
                matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, k);
                gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4], streamArray[4]);
                //gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4]);
            }
#pragma omp section
            {
                matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5], streamArray[5]);
                //gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5]);
            }
#pragma omp section
            {
                matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6], streamArray[6]);
                //gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6]);
            }
        }
    }

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

    for (int i = 0; i < 7; ++i) {
        cublasDestroy(handleArray[i]);
        cudaStreamDestroy(streamArray[i]);
    }
    free(handleArray);
    free(streamArray);
}

void gemmstrassen(float *C, float *A, float *B, int m, int k, int n, int ldA, int ldB, int ldC) {
    int mDiv2 = m / 2;
    int kDiv2 = k / 2;
    int nDiv2 = n / 2;

    float *A_11, *A_12, *A_21, *A_22;
    getSubmatrixPointer(A, mDiv2, kDiv2, ldA, A_11, A_12, A_21, A_22);

    float *B_11, *B_12, *B_21, *B_22;
    getSubmatrixPointer(B, kDiv2, nDiv2, ldB, B_11, B_12, B_21, B_22);

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

    cublasHandle_t *handleArray = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cublasCreate(handleArray + i);
    }

    cudaStream_t *streamArray = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cudaStreamCreate(streamArray + i);
    }

    int id;
#pragma omp parallel private(id)
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0], streamArray[0]);
                //gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0]);
            }

#pragma omp section
            {
                matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, ldB);
                gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1], streamArray[1]);
                //gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1]);
            }
#pragma omp section
            {
                matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, ldA);
                matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2], streamArray[2]);
                //gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2]);
            }
#pragma omp section
            {
                matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, ldA);
                matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3], streamArray[3]);
                //gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3]);
            }
#pragma omp section
            {
                matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, ldB);
                gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4], streamArray[4]);
                //gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4]);
            }
#pragma omp section
            {
                matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5], streamArray[5]);
                //gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5]);
            }
#pragma omp section
            {
                matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6], streamArray[6]);
                //gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6]);
            }
        }
    }

    float *C_11, *C_12, *C_21, *C_22;
    getSubmatrixPointer(C, mDiv2, nDiv2, ldC, C_11, C_12, C_21, C_22);

#pragma omp parallel
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(C_11, C_11, M1, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_11, C_11, M4, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_11, M1, M4, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
                //matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, ldC, ldC, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_21, C_21, M2, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_21, C_21, M4, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_21, M2, M4, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_12, C_12, M3, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_12, C_12, M5, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_12, M3, M5, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_22, C_22, M1, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixMinus(C_22, C_22, M2, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixMinus(C_22, M1, M2, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
                //matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, ldC, ldC, mDiv2);
            }
        }
    }

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

    for (int i = 0; i < 7; ++i) {
        cublasDestroy(handleArray[i]);
        cudaStreamDestroy(streamArray[i]);
    }
    free(handleArray);
    free(streamArray);
}

void gemmstrassenNOomp(float *C, float *A, float *B, int m, int k, int n, cublasHandle_t handle) {
    int mDiv2 = m / 2;
    int kDiv2 = k / 2;
    int nDiv2 = n / 2;

    float *A_11, *A_12, *A_21, *A_22;
    getSubmatrixPointer(A, mDiv2, kDiv2, m, A_11, A_12, A_21, A_22);

    float *B_11, *B_12, *B_21, *B_22;
    getSubmatrixPointer(B, kDiv2, nDiv2, k, B_11, B_12, B_21, B_22);

    float *M_1A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, m, m);
    float *M_1B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, k, k);
    float *M1 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_2A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, m, m);
    float *M_2B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, k);
    float *M2 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_3A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, m);
    float *M_3B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, k, k);
    float *M3 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_4A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, m);
    float *M_4B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, k, k);
    float *M4 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_5A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, m, m);
    float *M_5B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, k);
    float *M5 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_6A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, m, m);
    float *M_6B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, k, k);
    float *M6 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_7A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, m, m);
    float *M_7B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, k, k);
    float *M7 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *C_11, *C_12, *C_21, *C_22;
    getSubmatrixPointer(C, mDiv2, nDiv2, m, C_11, C_12, C_21, C_22);

    matrixAdd(C_11, M1, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);
    matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, m, m, mDiv2);
    matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, m, m, mDiv2);

    matrixAdd(C_21, M2, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);

    matrixAdd(C_12, M3, M5, mDiv2, nDiv2, m, mDiv2, mDiv2);

    matrixMinus(C_22, M1, M2, mDiv2, nDiv2, m, mDiv2, mDiv2);
    matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, m, m, mDiv2);
    matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, m, m, mDiv2);

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

void testGemm() {
    float *h_A = (float*)malloc(sizeof(float) * M * K);
    float *h_B = (float*)malloc(sizeof(float) * K * N);
    float *h_C = (float*)malloc(sizeof(float) * M * N);
    float *h_CTest = (float*)malloc(sizeof(float) * M * N);
    //float *h_CTest1 = (float*)malloc(sizeof(float) * M * N);

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);
    gemmcublas(h_CTest, h_A, h_B, M, K, N, handle);
    //gemm(h_CTest1, h_A, h_B, M, K, N);
    //test(h_CTest, h_CTest1, M, N);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    gemmstrassenNOomp(h_CTest, h_A, h_B, M, K, N, handle);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("gemmstarssenNOomp times %f ms\n", time);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    gemmstrassen(h_C, h_A, h_B, M, K, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("gemmstarssen times %f ms\n", time);
    test(h_C, h_CTest, M, N);

    int blockM = M / 4;
    int blockK = K / 4;
    int blockN = N / 4;
    blockMatrix bmatA(h_A, M, K, M, blockM, blockK);
    blockMatrix bmatB(h_B, K, N, K, blockK, blockN);
    blockMatrix bmatC(h_C, M, N, M, blockM, blockN);

    //std::vector<std::vector<int, int>> tasks;
    //for (int blockI = 0; blockI < bmatC.dimM; ++blockI) {
        //for (int blockJ = 0; blockJ < bmatC.dimN; ++blockJ) {
            //tasks.push_back(std::vector<int, int>{blockI, blockJ});
        //}
    //}

    memset(h_C, 0, sizeof(float) * M * N);
    time = 0.0f;


    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

#pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < bmatC.dimM; ++i) {
        for (int j = 0; j < bmatC.dimN; ++j) {
            for (int k = 0; k < bmatA.dimN; ++k) {
                printf("k:%d, i:%d, j:%d, ID:%d\n", k, i, j, omp_get_thread_num());
                gemmstrassen(bmatC.getBlockMatrix(i, j), bmatA.getBlockMatrix(i, k), bmatB.getBlockMatrix(k, j), blockM, blockK, blockN, M, K, M);
            }
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    printf("gemmstarssen block matrix times %f ms\n", time);
    test(h_C, h_CTest, M, N);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CTest);

    EXIT();
    return;
}

void testBlockmatrix() {
    float *A = (float*)malloc(sizeof(float) * M * K);
    initMatrix(A, M, K);

    int blockM = 1024;
    int blockN = 1024;
    blockMatrix bmatPtr(A, M, K, M, blockM, blockN);

    for (int blockI = 0; blockI < bmatPtr.dimM; ++blockI) {
        for (int blockJ = 0; blockJ < bmatPtr.dimN; ++blockJ) {
            int i = blockI * blockM;
            int j = blockJ * blockN;
            if (A[i + j * M] != *bmatPtr.pointers[blockI + blockJ * bmatPtr.dimM]) {
                printf("blockI: %d, blockJ: %d\n", blockI, blockJ);
            }
        }
    }

    EXIT();
    return;
}

int main() {
    testGemm();
    //testBlockmatrix();
    return 0;
}
