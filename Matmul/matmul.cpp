#include <iostream>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <vector>

#define HIP_CHECK_ERRORS(func) do { \
	hipError_t err = (func); \
	if (err != hipSuccess) { \
		std::cerr << "HIP error (" << __FILE__ << ":" << __LINE__ \
		<< ") :" << hipGetErrorString(err) << std::endl; \
		return -1; \
	} \
} while (0);\

void print_matrix(float *A, int M, int N){
	for (int i = 0; i < M; i++){
		std::cout << "[";
		for (int j = 0; j < N; j++){
			std::cout << A[i * N + j];
			if (j != N - 1){
				std::cout << ",";
			}
		}
		std::cout << "]\n";
	}
	std::cout << std::endl;
}

float* init_int_matrix(int M, int N, int max_int){
	float* matrix = new float[M * N];

	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			matrix[i * M + j] = (float)(rand() % max_int);
		}
	}
	return matrix;
}

__global__ void matmul_gpu(float *A, float *B, float *C, int M, int N, int K){
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < M && j < N){
		C[i * N + j] = 0.0;
		for (int k = 0; k < K; k++){
			C[i * N + j] += A[i * K + k] * B[k * N + j]; 
		}
	}
}

int main(){
	int M = 1;
	int N = 1;
	int K = 3;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid(((M + threadsPerBlock.y - 1) / threadsPerBlock.y), \
						(N + threadsPerBlock.x - 1) / threadsPerBlock.x);

	//float *h_A, *h_B;

	//h_A = init_int_matrix(M, K, M*K);
	//h_B = init_int_matrix(K, N, K*N);
	std::vector<float> h_A = {4.0, 5.0, 6.0};
	std::vector<float> h_B = {1.0, 2.0, 3.0};
	float *h_C = (float*)malloc(M * N * sizeof(float));

	print_matrix(h_A.data(), M, K);
	print_matrix(h_B.data(), K, N);

	float *d_A, *d_B, *d_C;

	HIP_CHECK_ERRORS(hipMalloc(&d_A, M * K * sizeof(float)));
	HIP_CHECK_ERRORS(hipMalloc(&d_B, K * N * sizeof(float)));
	HIP_CHECK_ERRORS(hipMalloc(&d_C, M * N * sizeof(float)));

	HIP_CHECK_ERRORS(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
	HIP_CHECK_ERRORS(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

	matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
	HIP_CHECK_ERRORS(hipDeviceSynchronize());

	HIP_CHECK_ERRORS(hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

	print_matrix(h_C, M, N);

	return 0;
}