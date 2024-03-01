#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "ped_model.h"

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

void Ped::Model::setupHeatmapCuda(){}

void Ped::Model::updateHeatmapCuda(){}

__global__ void fadeHeatmapCuda(){}

__global__ void intensifyHeatmapCuda(){}

__global__ void scaleHeatmapCuda(){}

__global__ void blurHeatmapCuda(){}