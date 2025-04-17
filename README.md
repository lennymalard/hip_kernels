# CUDA & HIP Programming Cheat Sheet (2025)

## Basics

### Compiler
- **CUDA**: `nvcc` (NVIDIA CUDA Compiler)
    ```bash
    nvcc program.cu -o program
    ```
    *Compiles CUDA source into executable*
    
- **HIP**: `hipcc` (HIP Compiler)
    ```bash
    hipcc program.cpp -o program
    ```
    *Compiles HIP source into executable*

### Memory Allocation
- **CUDA Device Memory Allocation**
    ```cpp
    int *d_array;
    cudaMalloc((void**)&d_array, size * sizeof(int));
    ```
    *Allocates memory on GPU device*
    - `d_array`: Pointer to store device memory address
    - `size * sizeof(int)`: Number of bytes to allocate

- **HIP Device Memory Allocation**
    ```cpp
    int *d_array;
    hipMalloc((void**)&d_array, size * sizeof(int));
    ```
    *Allocates memory on GPU device*
    - `d_array`: Pointer to store device memory address
    - `size * sizeof(int)`: Number of bytes to allocate

### Memory Copy
- **CUDA Host to Device**
    ```cpp
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);
    ```
    *Copies data from CPU to GPU memory*
    - `d_array`: Device memory destination pointer
    - `h_array`: Host memory source pointer
    - `size * sizeof(int)`: Number of bytes to copy
    - `cudaMemcpyHostToDevice`: Direction flag (CPU to GPU)
    
- **HIP Host to Device**
    ```cpp
    hipMemcpy(d_array, h_array, size * sizeof(int), hipMemcpyHostToDevice);
    ```
    *Copies data from CPU to GPU memory*
    - `d_array`: Device memory destination pointer
    - `h_array`: Host memory source pointer
    - `size * sizeof(int)`: Number of bytes to copy
    - `hipMemcpyHostToDevice`: Direction flag (CPU to GPU)

- **CUDA Device to Host**
    ```cpp
    cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    ```
    *Copies data from GPU to CPU memory*
    - `h_array`: Host memory destination pointer
    - `d_array`: Device memory source pointer
    - `size * sizeof(int)`: Number of bytes to copy
    - `cudaMemcpyDeviceToHost`: Direction flag (GPU to CPU)
    
- **HIP Device to Host**
    ```cpp
    hipMemcpy(h_array, d_array, size * sizeof(int), hipMemcpyDeviceToHost);
    ```
    *Copies data from GPU to CPU memory*
    - `h_array`: Host memory destination pointer
    - `d_array`: Device memory source pointer
    - `size * sizeof(int)`: Number of bytes to copy
    - `hipMemcpyDeviceToHost`: Direction flag (GPU to CPU)

### Free Device Memory
- **CUDA**
    ```cpp
    cudaFree(d_array);
    ```
    *Releases GPU memory*
    - `d_array`: Device pointer to memory that should be freed
    
- **HIP**
    ```cpp
    hipFree(d_array);
    ```
    *Releases GPU memory*
    - `d_array`: Device pointer to memory that should be freed

## CUDA & HIP Kernels

### Kernel Definition
- **CUDA**
    ```cpp
    __global__ void kernel_function(int *d_array) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        d_array[idx] = idx;
    }
    ```
    *Defines a GPU function executable by many threads*
    - `__global__`: Specifies function runs on device, callable from host
    - `d_array`: Device memory pointer accessible by all threads
    
- **HIP**
    ```cpp
    __global__ void kernel_function(int *d_array) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        d_array[idx] = idx;
    }
    ```
    *Defines a GPU function executable by many threads*
    - `__global__`: Specifies function runs on device, callable from host
    - `d_array`: Device memory pointer accessible by all threads

### Launching Kernels
- **CUDA**
    ```cpp
    kernel_function<<<num_blocks, threads_per_block>>>(d_array);
    ```
    *Executes kernel across specified grid of threads*
    - `num_blocks`: Number of blocks in the grid
    - `threads_per_block`: Number of threads in each block
    - `d_array`: Device memory pointer passed to kernel
    
- **HIP**
    ```cpp
    kernel_function<<<num_blocks, threads_per_block>>>(d_array);
    ```
    *Executes kernel across specified grid of threads*
    - `num_blocks`: Number of blocks in the grid
    - `threads_per_block`: Number of threads in each block
    - `d_array`: Device memory pointer passed to kernel

## Thread Hierarchy

- **CUDA**: 
    - `threadIdx.x`: Thread index within the block (x dimension)
    - `blockIdx.x`: Block index within the grid (x dimension)
    - `blockDim.x`: Number of threads per block (x dimension)
    - `gridDim.x`: Number of blocks in the grid (x dimension)
    ```cpp
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ```
    *Calculates global thread index*

- **HIP**: 
    - Identical to CUDA; `threadIdx`, `blockIdx`, `blockDim`, `gridDim` are used the same way.

## Memory Types

- **Global Memory**: Accessible by all threads, but slow
- **Shared Memory**: Fast, only accessible within the block
    ```cpp
    __shared__ int shared_mem[256];
    ```
    *Creates block-local shared memory array*
    - `shared_mem`: Array visible to all threads in the same block
    - `256`: Size of the shared memory array in elements
- **Constant Memory**: Read-only, cached, fast for small data
- **Texture Memory**: Optimized for 2D/3D access patterns

## Synchronization
- **Thread Synchronization**
    - **CUDA**
    ```cpp
    __syncthreads();
    ```
    *Synchronizes all threads in a block*
    - No arguments: Creates a barrier where all threads in a block must wait
    
    - **HIP**
    ```cpp
    __syncthreads();
    ```
    *Synchronizes all threads in a block*
    - No arguments: Creates a barrier where all threads in a block must wait

- **Device Synchronization**
    - **CUDA**
    ```cpp
    cudaDeviceSynchronize();
    ```
    *Waits for all GPU operations to complete*
    - No arguments: Blocks CPU execution until all GPU tasks finish
    
    - **HIP**
    ```cpp
    hipDeviceSynchronize();
    ```
    *Waits for all GPU operations to complete*
    - No arguments: Blocks CPU execution until all GPU tasks finish

## Error Handling
- **CUDA Error Checking**
    ```cpp
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    ```
    *Retrieves and displays last CUDA error*
    - `err`: Variable to store error code
    - `cudaGetLastError()`: Returns the last error that occurred
    - `cudaGetErrorString(err)`: Converts error code to human-readable string
    
- **HIP Error Checking**
    ```cpp
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(err));
    }
    ```
    *Retrieves and displays last HIP error*
    - `err`: Variable to store error code
    - `hipGetLastError()`: Returns the last error that occurred
    - `hipGetErrorString(err)`: Converts error code to human-readable string

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
    *Retrieves properties of the specified GPU device*
    - `deviceProp`: Structure to store device properties
    - `0`: Device ID (first GPU)
    - `deviceProp.totalGlobalMem`: Total global memory on device in bytes
    
- **HIP**
    ```cpp
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, 0);
    printf("Total memory: %ld MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    ```
    *Retrieves properties of the specified GPU device*
    - `deviceProp`: Structure to store device properties
    - `0`: Device ID (first GPU)
    - `deviceProp.totalGlobalMem`: Total global memory on device in bytes

### Multi-Device Programming
- **CUDA**
    ```cpp
    cudaSetDevice(device_id);
    ```
    *Switches active GPU to specified device*
    - `device_id`: Integer ID of the GPU to use (0-indexed)
    
- **HIP**
    ```cpp
    hipSetDevice(device_id);
    ```
    *Switches active GPU to specified device*
    - `device_id`: Integer ID of the GPU to use (0-indexed)

## Useful Commands

### List Available Devices
- **CUDA**
    ```bash
    nvidia-smi
    ```
    *Displays information about NVIDIA GPUs*
    
- **HIP**
    ```bash
    amd-smi
    ```
    *Lists available HIP-compatible devices*

### View Device Properties
- **CUDA**
    ```bash
    deviceQuery
    ```
    *Shows detailed CUDA device properties*
    
- **HIP**
    ```bash
    rocminfo
    ```
    *Shows detailed HIP device properties*

## Additional Resources

- **CUDA Toolkit Documentation**: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- **HIP Documentation**: [https://rocmdocs.amd.com/](https://rocmdocs.amd.com/)
- **CUDA Samples**: [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
- **HIP Samples**: [https://github.com/ROCm-Developer-Tools/hipify](https://github.com/ROCm-Developer-Tools/hipify)
