#include <iostream>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <time.h>

#define VECTOR_SIZE 100000000 

#define HIP_CHECK_WARNS(func) do { \
	hipError_t err = (func); \
	if (err != hipSuccess){ \
		std::cerr << "Error caught (" << __FILE__ << ":" << __LINE__ << "): " << hipGetErrorString(err) << std::endl; \
		return -1; \
	} \
} while (0); \

float get_time(){
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_int_vector(float *vec, int N, int max){
	for (int i = 0; i < N; i++){
		vec[i] = rand() % max+1;
	}
}

void init_float_vector(float *vec, int N){
	for (int i = 0; i < N; i++){
		vec[i] = (float)rand() / RAND_MAX;
	}
}

void print_vector(float *vec, int N){
	std::cout << "[";
	for (int i = 0; i < N; i++){
		std::cout << vec[i];
		if (i != N-1){
			std::cout << ",";
		}
	}
	std::cout << "]\n";
}

__global__ void add_vector_gpu(float *A, float *B, float *C, int N){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < N){
		C[idx] = A[idx] + B[idx];
	} 
}

void add_vector_cpu(float *A, float *B, float *C, int N){
	for (int i = 0; i < N; i++){
		C[i] = A[i] + B[i];
	}
}

int main(){
	int num_threads = 256;
	int num_blocks = (VECTOR_SIZE + num_threads - 1) / num_threads;

	float *h_A = (float*)malloc(VECTOR_SIZE * sizeof(float));
	float *h_B = (float*)malloc(VECTOR_SIZE * sizeof(float));
	float *h_C = (float*)malloc(VECTOR_SIZE * sizeof(float));

	init_float_vector(h_A, VECTOR_SIZE);
	init_float_vector(h_B, VECTOR_SIZE);

	//print_vector(h_A, VECTOR_SIZE);
	//print_vector(h_B, VECTOR_SIZE);

	printf("Benchmarking CPU vector addition...\n");
	float cpu_total_time = 0.0; 
	for (int i = 0; i < 20; i++){
		float ti = get_time();
		add_vector_cpu(h_A, h_B, h_C, VECTOR_SIZE);
		float tf = get_time();
		cpu_total_time += tf-ti;
	}
	float cpu_average_time = cpu_total_time / 20;
	printf("CPU average time: %f\n", cpu_average_time);
	
	//print_vector(h_C, VECTOR_SIZE);

	float *d_A, *d_B, *d_C;

	HIP_CHECK_WARNS(hipMalloc(&d_A, VECTOR_SIZE * sizeof(float)));
	HIP_CHECK_WARNS(hipMalloc(&d_B, VECTOR_SIZE * sizeof(float)));
	HIP_CHECK_WARNS(hipMalloc(&d_C, VECTOR_SIZE * sizeof(float)));

	HIP_CHECK_WARNS(hipMemcpy(d_A, h_A, VECTOR_SIZE * sizeof(float), hipMemcpyHostToDevice));
	HIP_CHECK_WARNS(hipMemcpy(d_B, h_B, VECTOR_SIZE * sizeof(float), hipMemcpyHostToDevice));

	printf("Benchmarking GPU vector addition...\n");
	float gpu_total_time = 0.0;
	for (int i = 0; i < 20; i++){
		float ti = get_time();
		add_vector_gpu<<<num_blocks, num_threads>>>(d_A, d_B, d_C, VECTOR_SIZE);
		HIP_CHECK_WARNS(hipDeviceSynchronize());
		float tf = get_time();
		gpu_total_time += tf-ti;
	}
	float gpu_average_time = gpu_total_time / 20;
	printf("GPU average time: %f\n", gpu_average_time);
	
	HIP_CHECK_WARNS(hipMemcpy(h_C, d_C, VECTOR_SIZE * sizeof(float), hipMemcpyDeviceToHost));
	//print_vector(d_C, VECTOR_SIZE);
	
	return 0;
}