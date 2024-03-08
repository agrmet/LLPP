// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

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
void Ped::Model::updateHeatmapSeq()
{
    // auto startTime = chrono::high_resolution_clock::now(); // Start timing

    auto startTime1 = chrono::high_resolution_clock::now();
	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			// heat fades
			heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
		}
	}
	auto endTime1 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> fadeTime = endTime1 - startTime1;

	auto startTime2 = chrono::high_resolution_clock::now();
	// Count how many agents want to go to each location
	for (int i = 0; i < agents.size(); i++)
	{
		Ped::Tagent* agent = agents[i];
		int x = agent->getDesiredX();
		int y = agent->getDesiredY();

		if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
		{
			continue;
		}

		// intensify heat for better color results
		heatmap[y][x] += 40;

	}

	for (int x = 0; x < SIZE; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
		}
	}
	auto endTime2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> intensifyTime = endTime2 - startTime2;

	auto startTime3 = chrono::high_resolution_clock::now();
	// Scale the data for visual representation
	for (int y = 0; y < SIZE; y++)
	{
		for (int x = 0; x < SIZE; x++)
		{
			int value = heatmap[y][x];
			for (int cellY = 0; cellY < CELLSIZE; cellY++)
			{
				for (int cellX = 0; cellX < CELLSIZE; cellX++)
				{
					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
				}
			}
		}
	}
	auto endTime3 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> scaleTime = endTime3 - startTime3;

	auto startTime4 = chrono::high_resolution_clock::now();
	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

#define WEIGHTSUM 273
	// Apply gaussian blurfilter		       
	for (int i = 2; i < SCALED_SIZE - 2; i++)
	{
		for (int j = 2; j < SCALED_SIZE - 2; j++)
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
		}
	}
	auto endTime4 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> blurTime = endTime4 - startTime4;

	printf("fadeTime: %f\n", fadeTime.count());
    printf("intensifyTime: %f\n", intensifyTime.count());
    printf("scaleTime: %f\n", scaleTime.count());
    printf("blurTime: %f\n", blurTime.count());

	// Stop timing and print duration
    // auto endTime = chrono::high_resolution_clock::now();
    // chrono::duration<double, std::milli> duration = endTime - startTime;
	// printf("updateHeatmapSeq timer: %f ms\n", duration.count());
}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}