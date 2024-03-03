#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "ped_model.h"
#include "heatmap_cuda.h"

// Initialize cuda streams, allocate memory on GPU and copy heatmap from CPU to GPU:
void Ped::Model::setupHeatmapCuda(){
    printf("hi");
    // Create a stream 'streamCuda':
    cudaStreamCreate(&streamCuda);

    // Allocate memory on GPU for heatmap, scaled_heatmap and blurred_heatmap:
    cudaMalloc(&heatmap_cuda, SIZE*SIZE*sizeof(int));    
    cudaMalloc(&scaled_heatmap_cuda, SCALED_SIZE*SCALED_SIZE*sizeof(int));
    cudaMalloc(&blurred_heatmap_cuda, SCALED_SIZE*SCALED_SIZE*sizeof(int));

    // Copy the data from CPU to GPU:
    cudaMemcpy(heatmap_cuda, heatmap[0], SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(scaled_heatmap_cuda, scaled_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(blurred_heatmap_cuda, blurred_heatmap[0], SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);

    // Some agent attributes we need for intensify function:
    cudaMalloc(&agentsDesiredX, agents.size()*sizeof(int));
    cudaMalloc(&agentsDesiredY, agents.size()*sizeof(int));

    // Temporary arrays for calculating where agents wants to go:
    desXsCPU = new int[agents.size()];
    desYsCPU = new int[agents.size()];

    // Add the desired x and y positions into the temporary arrays:
    for (int i = 0; i < agents.size(); i++){
        Ped::Tagent* agent = agents[i];
        desXsCPU[i] = agent->getDesiredX();
        desYsCPU[i] = agent->getDesiredY();
    }

    // Copy these from CPU to GPU:
    cudaMemcpy(agentsDesiredX, desXsCPU, agents.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(agentsDesiredY, desYsCPU, agents.size()*sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

// Call on kernels (functions) to update the heatmap:
void Ped::Model::updateHeatmapCuda(){
    // Call using the same stream (the calls needs to be executed in order)
    // Kernel execution configuration: <<<grid, block, shared, stream>>>
    // Fade: each thread gets 1 pixel to work on (grid = SIZE, block = SIZE, total threads = SIZE * SIZE):
    fadeHeatmapCuda<<<SIZE, SIZE, 0, streamCuda>>>(heatmap_cuda);
    // Intensify: each thread gets 1 agent to work on (1 block with #threads = #agents)
    intensifyHeatmapCuda<<<1, agents.size(), 0, streamCuda>>>(heatmap_cuda, agentsDesiredX, agentsDesiredY);
    // Scale: each thread scales 1 pixel
    scaleHeatmapCuda<<<SIZE, SIZE, 0, streamCuda>>>(heatmap_cuda, scaled_heatmap_cuda);
    blurHeatmapCuda<<<SIZE, SIZE, 0, streamCuda>>>(scaled_heatmap_cuda, blurred_heatmap_cuda);

    // Copy scaled heatmap data from GPU to CPU
	cudaMemcpy(blurred_heatmap[0], blurred_heatmap_cuda, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}

// Step 1: fade
__global__ void fadeHeatmapCuda(int *heatmap_cuda){
    // Calculate the unique index for each thread:
    // blockIdx.x = block index in grid
    // blockDim.x = size of a block (= SIZE)
    // threadIdx.x = thread index within block
    int threadIdxCuda = blockIdx.x * blockDim.x + threadIdx.x; // Represents a pixel on the map
    // Update the cuda-heatmap:
    heatmap_cuda[threadIdxCuda] = (int)round(heatmap_cuda[threadIdxCuda] * 0.80);
}

// Step 2: increment heat
__global__ void intensifyHeatmapCuda(int *heatmap_cuda, int *agentsDesiredX, int *agentsDesiredY){
    // Intensify heat by 40 on each desired position
    int threadIdxCuda = threadIdx.x;        // threadIdx here is one agent
    int desX = agentsDesiredX[threadIdxCuda];
    int desY = agentsDesiredY[threadIdxCuda];
    // Safe add if multiple agents wants to move to the same location:
    atomicAdd(&heatmap_cuda[desX + desY * SIZE], 40);
}

// Step 3: scale 
__global__ void scaleHeatmapCuda(int *heatmap_cuda, int *scaled_heatmap_cuda){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Flat index for 1D array

    int y = idx / SIZE; // Calculate y coordinate (row) in original heatmap
    int x = idx % SIZE; // Calculate x coordinate (column) in original heatmap

    // conversion from 2D to 1D:
    // index1D = row * rowWidth + column
    int value = heatmap_cuda[y * SIZE + x]; // Access the value

    // Same as in heatmap_seq:
    for (int cellY = 0; cellY < CELLSIZE; cellY++) {
            for (int cellX = 0; cellX < CELLSIZE; cellX++) {
                int scaledY = y * CELLSIZE + cellY;
                int scaledX = x * CELLSIZE + cellX;
                int scaledIdx = scaledY * SCALED_SIZE + scaledX; // calculate index in scaled heatmap
                scaled_heatmap_cuda[scaledIdx] = value;
            }
		}
}

// Step 4: apply gaussian blur filter
__global__ void blurHeatmapCuda(int *scaled_heatmap_cuda, int *blurred_heatmap_cuda){
    const int w[5][5] = {
			{ 1, 4, 7, 4, 1 },
			{ 4, 16, 26, 16, 4 },
			{ 7, 26, 41, 26, 7 },
			{ 4, 16, 26, 16, 4 },
			{ 1, 4, 7, 4, 1 }
    };
    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    // Determine the size of the shared memory area
    #define SHARED_SIZE (16 + 4)
    __shared__ int sharedMem[SHARED_SIZE][SHARED_SIZE];

    // Global index
    int col = bx + tx;
    int row = by + ty;

    // Load data into shared memory, including halo
    if (row < SCALED_SIZE && col < SCALED_SIZE) {
        sharedMem[ty + 2][tx + 2] = scaled_heatmap_cuda[row * SCALED_SIZE + col];
        
        // Load top and bottom halo
        if (ty < 2) {
            if (row >= 2) sharedMem[ty][tx + 2] = scaled_heatmap_cuda[(row - 2) * SCALED_SIZE + col];
            if (row < SCALED_SIZE - 2) sharedMem[ty + blockDim.y + 2][tx + 2] = scaled_heatmap_cuda[(row + 2) * SCALED_SIZE + col];
        }

        // Load left and right halo
        if (tx < 2) {
            if (col >= 2) sharedMem[ty + 2][tx] = scaled_heatmap_cuda[row * SCALED_SIZE + (col - 2)];
            if (col < SCALED_SIZE - 2) sharedMem[ty + 2][tx + blockDim.x + 2] = scaled_heatmap_cuda[row * SCALED_SIZE + (col + 2)];
        }
    }

    // Wait for all threads to finish loading into shared memory
    __syncthreads();

    // Apply the Gaussian blur using shared memory
    if (tx >= 2 && tx < blockDim.x - 2 && ty >= 2 && ty < blockDim.y - 2 && row >= 2 && row < SCALED_SIZE - 2 && col >= 2 && col < SCALED_SIZE - 2) {
        int sum = 0;
        for (int k = -2; k <= 2; k++) {
            for (int l = -2; l <= 2; l++) {
                sum += w[k + 2][l + 2] * sharedMem[ty + k + 2][tx + l + 2];
            }
        }
        int value = sum / WEIGHTSUM;
        blurred_heatmap_cuda[row * SCALED_SIZE + col] = 0x00FF0000 | value << 24;
    }
}

