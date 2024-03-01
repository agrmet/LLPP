// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// For CUDA implementation
#include <cuda_runtime.h>

// Constants for CUDA block sizes
#define BLOCK_CREATE 1024
#define BLOCK_SCALE 16
#define BLOCK_BLUR 32

// CUDA kernel for parallelizing the heatmap creation
__global__ void createHeatmapKernel(int* heatmap, int size, int* agentsData, int numAgents) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // Each thread handles one agent
    if (threadId < numAgents) {
        int agentX = agentsData[threadId * 2 + 1];
        int agentY = agentsData[threadId * 2 + 2];

        // Update heatmap for this agent's position
        if (agentX >= 0 && agentX < size && agentY >= 0 && agentY < size) {
            atomicAdd(&heatmap[agentX * size + agentY], 40);
        }
    }
}

// CUDA kernel for parallelizing the heatmap scaling
__global__ void scaleHeatmapKernel(int* heatmap, int size, int* scaled_heatmap, int cellsize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size) {
        int value = heatmap[y * size + x];
        for (int cellY = 0; cellY < cellsize; ++cellY) {
            for (int cellX = 0; cellX < cellsize; ++cellX) {
                scaled_heatmap[(y * cellsize + cellY) * size * cellsize + (x * cellsize + cellX)] = value;
            }
        }
    }
}

// CUDA kernel for parallelizing the heatmap blurring
__global__ void blurHeatmapKernel(int* scaled_heatmap, int* blurred_heatmap, int scaled_size) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};	
	#define WEIGHTSUM 273

    if (tx < scaled_size && ty < scaled_size) {
        // Apply gaussian blur filter
        int sum = 0;
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                int x = tx + j;
                int y = ty + i;

                // Adjust coordinates to ensure they are within bounds
                x = max(0, min(x, scaled_size - 1));
                y = max(0, min(y, scaled_size - 1));

                sum += w[i + 2][j + 2] * scaled_heatmap[y * scaled_size + x];
            }
        }

        int value = sum / WEIGHTSUM;
        blurred_heatmap[ty * scaled_size + tx] = 0x00FF0000 | value << 24;
    }
}

// Sets up the heatmap
void Ped::Model::setupHeatmapSeq()
{
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	heatmap = (int**)malloc(SIZE*sizeof(int*));

	scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

	for (int i = 0; i < SIZE; i++)
	{
		heatmap[i] = hm + SIZE*i;
	}
	for (int i = 0; i < SCALED_SIZE; i++)
	{
		scaled_heatmap[i] = shm + SCALED_SIZE*i;
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	}
}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq() {
    // Allocate GPU memory
    int* d_heatmap, * d_scaled_heatmap, * d_blurred_heatmap, * d_agents;
	int numAgents = agents.size();
    cudaMalloc((void**)&d_heatmap, SIZE * SIZE * sizeof(int));
    cudaMalloc((void**)&d_scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc((void**)&d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc((void**)&d_agents, (2 * numAgents) * sizeof(int));
	
	// Fade the heatmap with CPU
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
		}
	}

	// Transfer agent data (x and y) to GPU
    int* agentsData = new int[2 * numAgents];
    for (size_t i = 0; i < numAgents; ++i) {
        agentsData[i * 2] = agents[i]->getDesiredX();
        agentsData[i * 2 + 1] = agents[i]->getDesiredY();
    }

	// Copy agent and heatmap data to GPU
    cudaMemcpy(d_agents, agentsData, (2 * numAgents) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heatmap, heatmap[0], SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // CALL KERNEL: create the heatmap
    int numBlocks = (numAgents + BLOCK_CREATE - 1) / BLOCK_CREATE;
    dim3 gridSize(numBlocks, 1);
    createHeatmapKernel<<<gridSize, BLOCK_CREATE>>>(d_heatmap, SIZE, d_agents, numAgents);

    // Synchronize to ensure all CUDA operations are finished
    cudaDeviceSynchronize();

    // Copy heatmap from GPU to CPU
    cudaMemcpy(heatmap[0], d_heatmap, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// Adjust heatmap values to be max 255 with CPU
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
		}
	}

	// Copy adjusted heatmap to GPU
	cudaMemcpy(d_heatmap, heatmap[0], SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	// CALL KERNEL: scale the heatmap
	dim3 blockSizeScale(BLOCK_SCALE, BLOCK_SCALE);
    dim3 gridSizeScale((SIZE + blockSizeScale.x - 1) / blockSizeScale.x, (SIZE + blockSizeScale.y - 1) / blockSizeScale.y);
    scaleHeatmapKernel<<<gridSizeScale, blockSizeScale>>>(d_heatmap, SIZE, d_scaled_heatmap, CELLSIZE);
    
	// UNNECESSARY? Copy scaled heatmap data from GPU to CPU
	// cudaMemcpy(scaled_heatmap[0], d_scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	// OPTIONAL: Sync instead of copying data back to CPU
	cudaDeviceSynchronize();

	// CALL KERNEL: blur the heatmap
	dim3 blockSizeBlur(BLOCK_BLUR, BLOCK_BLUR); // Adjust block size for blur kernel
    dim3 gridSizeBlur((SCALED_SIZE + blockSizeBlur.x - 1) / blockSizeBlur.x, (SCALED_SIZE + blockSizeBlur.y - 1) / blockSizeBlur.y);
    blurHeatmapKernel<<<gridSizeBlur, blockSizeBlur>>>(d_scaled_heatmap, d_blurred_heatmap, SCALED_SIZE);
    
	// Copy scaled heatmap data from GPU to CPU
	cudaMemcpy(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// Free GPU memory
    cudaFree(d_heatmap);
    cudaFree(d_scaled_heatmap);
    cudaFree(d_blurred_heatmap);
    cudaFree(d_agents);
}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}
