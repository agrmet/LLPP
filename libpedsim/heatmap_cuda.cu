// Implements the heatmap functionality with CUDA
#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// For CUDA implementation
#include <cuda_runtime.h>

// Constant for CUDA
#define BLOCK_SIZE 32
#define TILE_SIZE 32

// CUDA kernel for the heatmap fading
__global__ void fadeHeatmap(int* d_heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < SIZE && y < SIZE) {
        d_heatmap[y * SIZE + x] = (int)round(d_heatmap[y * SIZE + x] * 0.80);
    }
}

// CUDA kernel for intensifying heatmap
__global__ void intensifyHeatmap(int* d_heatmap, int* agentsX, int* agentsY, int numAgents) {
    int agentId = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one agent
    if (agentId < numAgents) {
        int x = agentsX[agentId];
        int y = agentsY[agentId];

        // Update heatmap for this agent's position
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
            int oldValue = atomicAdd(&d_heatmap[y * SIZE + x], 40);
            
            // Cap the value at 255
            if (oldValue + 40 > 255) {
                atomicExch(&d_heatmap[y * SIZE + x], 255);
            }
        }
    }
}

// CUDA kernel for the heatmap scaling
__global__ void scaleHeatmap(int* d_heatmap, int* scaled_heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < SIZE && y < SIZE) {
        int value = d_heatmap[y * SIZE + x];

        for (int cellY = 0; cellY < CELLSIZE; cellY++) {
            for (int cellX = 0; cellX < CELLSIZE; cellX++) {
                scaled_heatmap[(y * CELLSIZE + cellY) * SIZE * CELLSIZE + (x * CELLSIZE + cellX)] = value;
            }
        }
    }
}

// CUDA kernel for the heatmap blurring
__global__ void blurHeatmap(int* scaled_heatmap, int* blurred_heatmap) {
    __shared__ int shared_data[TILE_SIZE + 4][TILE_SIZE + 4]; // Allocate shared memory tile
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute indices for reading and writing to shared memory
    int sharedX = threadIdx.x + 2;
    int sharedY = threadIdx.y + 2;

    // Load data from global memory into shared memory
    if (x < SCALED_SIZE && y < SCALED_SIZE) {
        shared_data[sharedY][sharedX] = scaled_heatmap[y * SCALED_SIZE + x];
    }
    // Handle border cases by loading additional data into shared memory
    if (threadIdx.x < 2) {
        if (x - 2 >= 0) {
            shared_data[sharedY][threadIdx.x] = scaled_heatmap[y * SCALED_SIZE + (x - 2)];
        }
        if (x + blockDim.x < SCALED_SIZE) {
            shared_data[sharedY][threadIdx.x + blockDim.x + 2] = scaled_heatmap[y * SCALED_SIZE + (x + blockDim.x)];
        }
    }
    if (threadIdx.y < 2) {
        if (y - 2 >= 0) {
            shared_data[threadIdx.y][sharedX] = scaled_heatmap[(y - 2) * SCALED_SIZE + x];
        }
        if (y + blockDim.y < SCALED_SIZE) {
            shared_data[threadIdx.y + blockDim.y + 2][sharedX] = scaled_heatmap[(y + blockDim.y) * SCALED_SIZE + x];
        }
    }

    __syncthreads(); // All data loaded into shared memory

    // Weights for blur filter
    const int w[5][5] = {
        { 1, 4, 7, 4, 1 },
        { 4, 16, 26, 16, 4 },
        { 7, 26, 41, 26, 7 },
        { 4, 16, 26, 16, 4 },
        { 1, 4, 7, 4, 1 }
    };
    int weightsum = 273;

    if (x < SCALED_SIZE && y < SCALED_SIZE) {
        // Apply gaussian blur filter using data from shared memory
        int sum = 0;
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                sum += w[i + 2][j + 2] * shared_data[sharedY + i][sharedX + j];
            }
        }

        int value = sum / weightsum;
        blurred_heatmap[y * SCALED_SIZE + x] = 0x00FF0000 | value << 24;
    }
}

void Ped::Model::setupHeatmapCUDA()
{
    // Allocate GPU memory
    cudaMalloc((void**)&d_heatmap, SIZE * SIZE * sizeof(int));
    cudaMalloc((void**)&d_scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc((void**)&d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc((void**)&d_agents_x, agents.size() * sizeof(int));
    cudaMalloc((void**)&d_agents_y, agents.size() * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_heatmap, heatmap[0], SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaled_heatmap, scaled_heatmap[0], SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blurred_heatmap, blurred_heatmap[0], SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
}

void Ped::Model::updateHeatmapCUDA() {
	int numAgents = agents.size();
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((SIZE + blockSize.x - 1) / blockSize.x, (SIZE + blockSize.y - 1) / blockSize.y);
	
    // CALL KERNEL: fade the heatmap
    fadeHeatmap<<<gridSize, blockSize>>>(d_heatmap);

	// Transfer agent data (x and y) to GPU
    int* agentsX = new int[numAgents];
    int* agentsY = new int[numAgents];
    for (int i = 0; i < numAgents; i++) {
        agentsX[i] = agents[i]->getDesiredX();
        agentsY[i] = agents[i]->getDesiredY();
    }
    cudaMemcpy(d_agents_x, agentsX, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_y, agentsY, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    
    // CALL KERNEL: intensify the heatmap (1 dim block in this implementation)
    cudaDeviceSynchronize();
    dim3 gridSizeIntensify((numAgents + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    intensifyHeatmap<<<gridSizeIntensify, BLOCK_SIZE>>>(d_heatmap, d_agents_x, d_agents_y, numAgents);

    // CALL KERNEL: scale the heatmap
    cudaDeviceSynchronize();
	scaleHeatmap<<<gridSize, blockSize>>>(d_heatmap, d_scaled_heatmap);
    
	// CALL KERNEL: blur the heatmap
	blurHeatmap<<<gridSize, blockSize>>>(d_scaled_heatmap, d_blurred_heatmap);
    
	// Copy blurred heatmap data from GPU to CPU
	cudaMemcpy(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}