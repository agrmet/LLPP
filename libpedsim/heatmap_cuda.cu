#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "ped_model.h"
#include "heatmap_cuda.h"

// Initialize cuda streams, allocate memory on GPU and copy heatmap from CPU to GPU:
void Ped::Model::setupHeatmapCuda(){
    // Create a stream 'streamCuda':
    cudaStreamCreate(&streamCuda);

    // Allocate memory on GPU for heatmap, scaled_heatmap and blurred_heatmap:
    cudaMalloc(&heatmap_cuda, SIZE*SIZE*sizeof(int));    
    cudaMalloc(&scaled_heatmap_cuda, SCALED_SIZE*SCALED_SIZE*sizeof(int));
    cudaMalloc(&blurred_heatmap_cuda, SCALED_SIZE*SCALED_SIZE*sizeof(int));

    // Copy the data from CPU to GPU:
    cudaMemcpy(heatmap_cuda, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(scaled_heatmap_cuda, scaled_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(blurred_heatmap_cuda, blurred_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);

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
}

// Call on kernels (functions) to update the heatmap:
void Ped::Model::updateHeatmapCuda(){
    // Call using the same stream (the calls needs to be executed in order)
    // Kernel execution configuration: <<<grid, block, shared, stream>>>
    // Fade: each thread gets 1 pixel to work on (grid = SIZE, block = SIZE, total threads = SIZE * SIZE):
    fadeHeatmapCuda<<<SIZE, SIZE, 0, streamCuda>>>(heatmap_cuda);
    // Intensify: each thread gets 1 agent to work on (1 block with #threads = #agents)
    intensifyHeatmapCuda<<<1, agents.size(), 0, streamCuda>>>(heatmap_cuda, agentsDesiredX, agentsDesiredY);
    scaleHeatmapCuda;
    blurHeatmapCuda;
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
__global__ void scaleHeatmapCuda(){
}

// Step 4: apply gaussian blur filter
__global__ void blurHeatmapCuda(){
    const int w[5][5] = {
			{ 1, 4, 7, 4, 1 },
			{ 4, 16, 26, 16, 4 },
			{ 7, 26, 41, 26, 7 },
			{ 4, 16, 26, 16, 4 },
			{ 1, 4, 7, 4, 1 }
	};

}