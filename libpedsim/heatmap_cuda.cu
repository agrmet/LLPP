// Implements the heatmap functionality with CUDA
#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// For CUDA implementation
#include <cuda_runtime.h>

// Constant for CUDA
#define BLOCK_SIZE 32

// CUDA kernel for the heatmap fading
__global__ void fadeHeatmap(int* d_heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
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

    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
        int value = d_heatmap[y * SIZE + x];

        for (int cellY = 0; cellY < CELLSIZE; cellY++) {
            for (int cellX = 0; cellX < CELLSIZE; cellX++) {
                scaled_heatmap[(y * CELLSIZE + cellY) * SCALED_SIZE + (x * CELLSIZE + cellX)] = value;
            }
        }
    }
}

// CUDA kernel for the heatmap blurring
__global__ void blurHeatmap(int* scaled_heatmap, int* blurred_heatmap) {
    // Shared mem for the block, TILE has 2 extra cells on each side
    __shared__ int shared_data[BLOCK_SIZE + 4][BLOCK_SIZE + 4];
   
    // Position of this thread in the grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Position for shared mem (offset by 2 to handle border cases)
    int sharedX = threadIdx.x + 2;
    int sharedY = threadIdx.y + 2;

    // Load data from scaled_heatmap into shared mem
    if (x >= 0 && x < SCALED_SIZE && y >= 0 && y < SCALED_SIZE) {
        shared_data[sharedY][sharedX] = scaled_heatmap[y * SCALED_SIZE + x];
    }
    // If this thread is close to the block borders, load extra data into shared mem
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

    __syncthreads(); // Wait for all data to be loaded into shared mem

    // Weights for blur filter
    const int w[5][5] = {
        { 1, 4, 7, 4, 1 },
        { 4, 16, 26, 16, 4 },
        { 7, 26, 41, 26, 7 },
        { 4, 16, 26, 16, 4 },
        { 1, 4, 7, 4, 1 }
    };
    int weightsum = 273;

    if (x >= 0 && x < SCALED_SIZE && y >= 0 && y < SCALED_SIZE) {
        // Apply blur filter using data from shared mem
        int sum = 0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
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
    
    // No need to copy these to GPU (?!), as they are empty at this point:
    // cudaMemcpy(d_scaled_heatmap, scaled_heatmap[0], SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_blurred_heatmap, blurred_heatmap[0], SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
}

void Ped::Model::updateHeatmapCUDA() {
    // auto startTime = chrono::high_resolution_clock::now(); // Start timing
    
	int numAgents = agents.size();
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((SIZE + blockSize.x - 1) / blockSize.x, (SIZE + blockSize.y - 1) / blockSize.y);
	
    auto startTime0 = chrono::high_resolution_clock::now();
	// Transfer agent data (x and y) to GPU
    int *agentsX = new int[numAgents];
    int *agentsY = new int[numAgents];
    for (int i = 0; i < numAgents; i++) {
        agentsX[i] = agents[i]->getDesiredX();
        agentsY[i] = agents[i]->getDesiredY();
    }
    cudaMemcpy(d_agents_x, agentsX, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_y, agentsY, numAgents * sizeof(int), cudaMemcpyHostToDevice);
	auto endTime0 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> memcpyTime = endTime0 - startTime0;

    // CALL KERNEL: fade the heatmap
    auto startTime1 = chrono::high_resolution_clock::now();
    fadeHeatmap<<<gridSize, blockSize>>>(d_heatmap);
    cudaDeviceSynchronize();
	auto endTime1 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> fadeTime = endTime1 - startTime1;
    
    // CALL KERNEL: intensify the heatmap (1 dim block in this implementation)
    auto startTime2 = chrono::high_resolution_clock::now();
    dim3 gridSizeIntensify((numAgents + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    intensifyHeatmap<<<gridSizeIntensify, BLOCK_SIZE>>>(d_heatmap, d_agents_x, d_agents_y, numAgents);
    cudaDeviceSynchronize();
	auto endTime2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> intensifyTime = endTime2 - startTime2;

    // CALL KERNEL: scale the heatmap
    auto startTime3 = chrono::high_resolution_clock::now();
	scaleHeatmap<<<gridSize, blockSize>>>(d_heatmap, d_scaled_heatmap);
	auto endTime3 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> scaleTime = endTime3 - startTime3;

	// CALL KERNEL: blur the heatmap
    auto startTime4 = chrono::high_resolution_clock::now();
	blurHeatmap<<<gridSize, blockSize>>>(d_scaled_heatmap, d_blurred_heatmap);
	auto endTime4 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> blurTime = endTime4 - startTime4;

	// Copy blurred heatmap data from GPU to CPU
    auto startTime5 = chrono::high_resolution_clock::now();
	cudaMemcpy(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	auto endTime5 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> memcpyTime2 = endTime5 - startTime5;

    // // Stop timing and print duration
    // auto endTime = chrono::high_resolution_clock::now();
    // chrono::duration<double, std::milli> duration = endTime - startTime;
	// printf("updateHeatmapCUDA timer: %f ms\n", duration.count());

    printf("memcpyTime: %f\n", memcpyTime.count() + memcpyTime2.count());
    printf("fadeTime: %f\n", fadeTime.count());
    printf("intensifyTime: %f\n", intensifyTime.count());
    printf("scaleTime: %f\n", scaleTime.count());
    printf("blurTime: %f\n", blurTime.count());
}