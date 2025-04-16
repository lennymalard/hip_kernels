#include <iostream>
#include <hip/hip_runtime.h>

__constant__ char d_message[20];

__global__ void welcome(char* msg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    msg[idx] = d_message[idx];
}

int main() {
    char* d_msg;
    char* h_msg;
    const char message[] = "Welcome to LeetGPU!";
    const int length = strlen(message) + 1;

    // Allocate host and device memory
    h_msg = (char*)malloc(length * sizeof(char));
    hipMalloc(&d_msg, length * sizeof(char));
    
    // Copy message to constant memory
    hipMemcpyToSymbol(d_message, message, length);
    
    // Launch welcome kernel
    welcome<<<1, length>>>(d_msg);
    
    // Copy result back to host
    hipMemcpy(h_msg, d_msg, length * sizeof(char), hipMemcpyDeviceToHost);
    h_msg[length-1] = '\0';

    std::cout << h_msg << "\n";
    
    // Cleanup
    free(h_msg);
    hipFree(d_msg);
    
    return 0;
}
