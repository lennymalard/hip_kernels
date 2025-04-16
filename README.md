# CUDA & HIP Programming Cheat Sheet (2025)

## Basics

### Compiler
- **CUDA**: `nvcc` (NVIDIA CUDA Compiler)
    ```bash
    nvcc program.cu -o program
    ```
- **HIP**: `hipcc` (HIP Compiler)
    ```bash
    hipcc program.cpp -o program
    ```

### Memory Allocation
- **CUDA Device Memory Allocation**
    ```cpp
    int *d_array;
    cudaMalloc((void**)&d_array, size * sizeof(int));
    ```
- **HIP Device Memory Allocation**
    ```cpp
    int *d_array;
    hipMalloc((void**)&d_array, size * sizeof(int));
    ```

### Memory Copy
- **CUDA Host to Device**
    ```cpp
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);
    ```
- **HIP Host to Device**
    ```cpp
    hipMemcpy(d_array, h_array, size * sizeof(int), hipMemcpyHostToDevice);
    ```

- **CUDA Device to Host**
    ```cpp
    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    ```
- **HIP Device to Host**
    ```cpp
    hipMemcpy(h_array, d_array, size * sizeof(int), hipMemcpyDeviceToHost);
    ```

### Free Device Memory
- **CUDA**
    ```cpp
    cudaFree(d_array);
    ```
- **HIP**
    ```cpp
    hipFree(d_array);
    ```

## CUDA & HIP Kernels

### Kernel Definition
- **CUDA**
    ```cpp
    __global__ void kernel_function(int *d_array) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        d_array[idx] = idx;
    }
    ```
- **HIP**
    ```cpp
    __global__ void kernel_function(int *d_array) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        d_array[idx] = idx;
    }
    ```

### Launching Kernels
- **CUDA**
    ```cpp
    kernel_function<<<num_blocks, threads_per_block>>>(d_array);
    ```
- **HIP**
    ```cpp
    kernel_function<<<num_blocks, threads_per_block>>>(d_array);
    ```

## Thread Hierarchy

- **CUDA**: 
    - `threadIdx.x`: Thread index within the block
    - `blockIdx.x`: Block index within the grid
    - `blockDim.x`: Number of threads per block
    - `gridDim.x`: Number of blocks in the grid
    ```cpp
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ```

- **HIP**: 
    - Identical to CUDA; `threadIdx`, `blockIdx`, `blockDim`, `gridDim` are used the same way.

## Memory Types

- **Global Memory**: Accessible by all threads, but slow
- **Shared Memory**: Fast, only accessible within the block
    ```cpp
    __shared__ int shared_mem[256];
    ```
- **Constant Memory**: Read-only, cached, fast for small data
- **Texture Memory**: Optimized for 2D/3D access patterns

## Synchronization
- **Thread Synchronization**
    - **CUDA**
    ```cpp
    __syncthreads();
    ```
    - **HIP**
    ```cpp
    __syncthreads();
    ```

- **Device Synchronization**
    - **CUDA**
    ```cpp
    cudaDeviceSynchronize();
    ```
    - **HIP**
    ```cpp
    hipDeviceSynchronize();
    ```

## Error Handling
- **CUDA Error Checking**
    ```cpp
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    ```
- **HIP Error Checking**
    ```cpp
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(err));
    }
    ```

## Performance Optimization
- **Memory Coalescing**: Access memory in a coalesced manner to improve performance
- **Occupancy**: Maximize the number of threads per block to utilize the GPU resources effectively
- **Shared Memory Usage**: Use shared memory to avoid global memory accesses

## Miscellaneous

### Get Device Properties
- **CUDA**
    ```cpp
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Total memory: %ld MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    ```
- **HIP**
    ```cpp
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 0);
    printf("Total memory: %ld MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    ```

### Multi-Device Programming
- **CUDA**
    ```cpp
    cudaSetDevice(device_id);  // Switch between devices
    ```
- **HIP**
    ```cpp
    hipSetDevice(device_id);  // Switch between devices
    ```

## Useful Commands

### List Available Devices
- **CUDA**
    ```bash
    nvidia-smi
    ```
- **HIP**
    ```bash
    hipDeviceQuery
    ```

### View Device Properties
- **CUDA**
    ```bash
    deviceQuery
    ```
- **HIP**
    ```bash
    hipDeviceQuery
    ```

## Additional Resources

- **CUDA Toolkit Documentation**: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- **HIP Documentation**: [https://rocmdocs.amd.com/](https://rocmdocs.amd.com/)
- **CUDA Samples**: [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
- **HIP Samples**: [https://github.com/ROCm-Developer-Tools/hipify](https://github.com/ROCm-Developer-Tools/hipify)
