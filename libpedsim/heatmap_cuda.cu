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

// Constants for CUDA block sizes
#define BLOCK_CREATE 1024
#define BLOCK_SCALE 16
#define BLOCK_BLUR 32

// CUDA kernel for parallelizing the heatmap creation
__global__ void createHeatmapKernel(int* heatmap, int size, int* agentsXY, int numAgents) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // Each thread handles one agent
    if (threadId < numAgents) {
        int agenthreadX = agentsXY[threadId * 2 + 1];
        int agenthreadY = agentsXY[threadId * 2 + 2];

        // Update heatmap for this agent's position
        if (agenthreadX >= 0 && agenthreadX < size && agenthreadY >= 0 && agenthreadY < size) {
            atomicAdd(&heatmap[agenthreadX * size + agenthreadY], 40);
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
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};	
	int weightsum = 273;

    if (threadX < scaled_size && threadY < scaled_size) {
        // Apply gaussian blur filter
        int sum = 0;
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                int x = threadX + j;
                int y = threadY + i;

                // Adjust coordinates to ensure they are within bounds
                x = max(0, min(x, scaled_size - 1));
                y = max(0, min(y, scaled_size - 1));

                sum += w[i + 2][j + 2] * scaled_heatmap[y * scaled_size + x];
            }
        }

        int value = sum / weightsum;
        blurred_heatmap[threadY * scaled_size + threadX] = 0x00FF0000 | value << 24;
    }
}

// CUDA kernel for parallelizing the heatmap blurring with shared memory
// __global__ void blurHeatmapKernel(int* scaled_heatmap, int* blurred_heatmap, int scaled_size) {
//     __shared__ int sharedMem[BLOCK_BLUR + 4][BLOCK_BLUR + 4]; // Allocate shared memory with padding

//     int threadX = threadIdx.x;
//     int threadY = threadIdx.y;
//     int x = blockIdx.x * blockDim.x + threadX;
//     int y = blockIdx.y * blockDim.y + threadY;

//     // Weights for blur filter
// 	const int w[5][5] = {
// 		{ 1, 4, 7, 4, 1 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 7, 26, 41, 26, 7 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 1, 4, 7, 4, 1 }
// 	};	
// 	int weightsum = 273;

//     // Load data into shared memory with padding
//     if (x < scaled_size && y < scaled_size) {
//         sharedMem[threadY + 2][threadX + 2] = scaled_heatmap[y * scaled_size + x];

//         if (threadX < 2) {
//             sharedMem[threadY + 2][threadX] = scaled_heatmap[y * scaled_size + max(0, x - 2)];
//             sharedMem[threadY + 2][threadX + blockDim.x + 2] = scaled_heatmap[y * scaled_size + min(scaled_size - 1, x + blockDim.x)];
//         }

//         if (threadY < 2) {
//             sharedMem[threadY][threadX + 2] = scaled_heatmap[max(0, y - 2) * scaled_size + x];
//             sharedMem[threadY + blockDim.y + 2][threadX + 2] = scaled_heatmap[min(scaled_size - 1, y + blockDim.y) * scaled_size + x];
//         }

//         if (threadX < 2 && threadY < 2) {
//             sharedMem[threadY][threadX] = scaled_heatmap[max(0, y - 2) * scaled_size + max(0, x - 2)];
//             sharedMem[threadY][threadX + blockDim.x + 2] = scaled_heatmap[max(0, y - 2) * scaled_size + min(scaled_size - 1, x + blockDim.x)];
//             sharedMem[threadY + blockDim.y + 2][threadX] = scaled_heatmap[min(scaled_size - 1, y + blockDim.y) * scaled_size + max(0, x - 2)];
//             sharedMem[threadY + blockDim.y + 2][threadX + blockDim.x + 2] = scaled_heatmap[min(scaled_size - 1, y + blockDim.y) * scaled_size + min(scaled_size - 1, x + blockDim.x)];
//         }
//     } else {
//         // Padding for out-of-bounds indices
//         sharedMem[threadY + 2][threadX + 2] = 0;

//         if (threadX < 2) {
//             sharedMem[threadY + 2][threadX] = 0;
//             sharedMem[threadY + 2][threadX + blockDim.x + 2] = 0;
//         }

//         if (threadY < 2) {
//             sharedMem[threadY][threadX + 2] = 0;
//             sharedMem[threadY + blockDim.y + 2][threadX + 2] = 0;
//         }

//         if (threadX < 2 && threadY < 2) {
//             sharedMem[threadY][threadX] = 0;
//             sharedMem[threadY][threadX + blockDim.x + 2] = 0;
//             sharedMem[threadY + blockDim.y + 2][threadX] = 0;
//             sharedMem[threadY + blockDim.y + 2][threadX + blockDim.x + 2] = 0;
//         }
//     }

//     __syncthreads();

//     // Apply blur filter
//     if (x < scaled_size && y < scaled_size) {
//         int sum = 0;
//         for (int i = -2; i <= 2; ++i) {
//             for (int j = -2; j <= 2; ++j) {
//                 sum += w[i + 2][j + 2] * sharedMem[threadY + 2 + i][threadX + 2 + j];
//             }
//         }
//         int value = sum / weightsum;
//         blurred_heatmap[y * scaled_size + x] = 0x00FF0000 | value << 24;
//     }
// }

void Ped::Model::updateHeatmapCUDA() {
    // Allocate GPU memory
    int* d_heatmap, * d_scaled_heatmap, * d_blurred_heatmap, * d_agents_xy;
	int numAgents = agents.size();
    cudaMalloc((void**)&d_heatmap, SIZE * SIZE * sizeof(int));
    cudaMalloc((void**)&d_scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc((void**)&d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc((void**)&d_agents_xy, (2 * numAgents) * sizeof(int));
	
	// Fade the heatmap with CPU
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
		}
	}

	// Transfer agent data (x and y) to GPU
    int* agentsXY = new int[2 * numAgents];
    for (size_t i = 0; i < numAgents; ++i) {
        agentsXY[i * 2] = agents[i]->getDesiredX();
        agentsXY[i * 2 + 1] = agents[i]->getDesiredY();
    }

	// Copy agent and heatmap data to GPU
    cudaMemcpy(d_agents_xy, agentsXY, (2 * numAgents) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heatmap, heatmap[0], SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // CALL KERNEL: create the heatmap
    int numBlocks = (numAgents + BLOCK_CREATE - 1) / BLOCK_CREATE;
    dim3 gridSize(numBlocks, 1);
    createHeatmapKernel<<<gridSize, BLOCK_CREATE>>>(d_heatmap, SIZE, d_agents_xy, numAgents);

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
    cudaFree(d_agents_xy);
}