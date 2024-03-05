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

// Constants for CUDA block sizes (32 * 32 = 1024 threads per block)
#define BLOCK_SCALE 32

// CUDA kernel for the heatmap fading
__global__ void fadeHeatmap(int* heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < SIZE && y < SIZE) {
        heatmap[y * SIZE + x] = (int)round(heatmap[y * SIZE + x] * 0.80);
    }
}

// CUDA kernel for intensifying heatmap
__global__ void intensifyHeatmap(int* heatmap, int* agentsX, int* agentsY, int numAgents) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // Each thread handles one agent
    if (threadId < numAgents) {
        int agentX = agentsX[threadId];
        int agentY = agentsY[threadId];

        // Update heatmap for this agent's position
        if (agentX >= 0 && agentX < SIZE && agentY >= 0 && agentY < SIZE) {
            atomicAdd(&heatmap[agentX * SIZE + agentY], 40);

            // int oldValue = atomicAdd(&heatmap[agentX * SIZE + agentY], 40);
            // if (oldValue + 40 > 255) {
            //     atomicExch(&heatmap[agentX * SIZE + agentY], 255);
            // }
        }
    }
}

// CUDA kernel for the heatmap adjustment
__global__ void adjustHeatmap(int* heatmap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < SIZE && y < SIZE) {
        heatmap[y * SIZE + x] = heatmap[y * SIZE + x] < 255 ? heatmap[y * SIZE + x] : 255;
    }
}

// CUDA kernel for the heatmap scaling
__global__ void scaleHeatmap(int* heatmap, int* scaled_heatmap, int cellsize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < SIZE && y < SIZE) {
        int value = heatmap[y * SIZE + x];

        for (int cellY = 0; cellY < cellsize; cellY++) {
            for (int cellX = 0; cellX < cellsize; cellX++) {
                scaled_heatmap[(y * cellsize + cellY) * SIZE * cellsize + (x * cellsize + cellX)] = value;
            }
        }
    }
}

// CUDA kernel for the heatmap blurring
__global__ void blurHeatmap(int* scaled_heatmap, int* blurred_heatmap, int scaled_size) {
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
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
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

// CUDA kernel for the heatmap blurring with shared memory
// __global__ void blurHeatmap(int* scaled_heatmap, int* blurred_heatmap, int scaled_size) {
//     __shared__ int sharedMem[BLOCK_SCALE + 4][BLOCK_SCALE + 4]; // Allocate shared memory with padding

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
//         for (int i = -2; i <= 2; i++) {
//             for (int j = -2; j <= 2; j++) {
//                 sum += w[i + 2][j + 2] * sharedMem[threadY + 2 + i][threadX + 2 + j];
//             }
//         }
//         int value = sum / weightsum;
//         blurred_heatmap[y * scaled_size + x] = 0x00FF0000 | value << 24;
//     }
// }

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
    dim3 blockSizeScale(BLOCK_SCALE, BLOCK_SCALE);
    dim3 gridSizeScale((SIZE + blockSizeScale.x - 1) / blockSizeScale.x, (SIZE + blockSizeScale.y - 1) / blockSizeScale.y);
	
    // CALL KERNEL: fade the heatmap
    fadeHeatmap<<<gridSizeScale, blockSizeScale>>>(d_heatmap);

	// Transfer agent data (x and y) to GPU
    int* agentsX = new int[numAgents];
    int* agentsY = new int[numAgents];
    for (int i = 0; i < numAgents; i++) {
        agentsX[i] = agents[i]->getDesiredX();
        agentsY[i] = agents[i]->getDesiredY();
    }

	// Copy agent data to GPU
    cudaMemcpy(d_agents_x, agentsX, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agents_y, agentsY, numAgents * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    // CALL KERNEL: create the heatmap
    dim3 gridSize(((numAgents + SIZE - 1) / SIZE), 1);
    intensifyHeatmap<<<gridSize, SIZE>>>(d_heatmap, d_agents_x, d_agents_y, numAgents);

    cudaDeviceSynchronize();

    // ADJUSTING HEATMAP VALUES, NOT NEEDED?
    // adjustHeatmap<<<gridSizeScale, blockSizeScale>>>(d_heatmap);
    // cudaDeviceSynchronize();

	// CALL KERNEL: scale the heatmap
	scaleHeatmap<<<gridSizeScale, blockSizeScale>>>(d_heatmap, d_scaled_heatmap, CELLSIZE);
    
	cudaDeviceSynchronize();

	// CALL KERNEL: blur the heatmap
	blurHeatmap<<<gridSizeScale, blockSizeScale>>>(d_scaled_heatmap, d_blurred_heatmap, SCALED_SIZE);
    
	// Copy scaled heatmap data from GPU to CPU
	cudaMemcpy(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}