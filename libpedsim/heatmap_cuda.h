#include "cuda_runtime.h"

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE *CELLSIZE
#define WEIGHTSUM 273

// For heatmapCuda: 
// heatmap_cuda is a 1D representation of the 2D heatmap.


void setupHeatmapCuda();
void updateHeatmapCuda();

__global__ void fadeHeatmapCuda(int *heatmap_cuda);
__global__ void intensifyHeatmapCuda(int *heatmap_cuda, int *agentsDesiredX, int *agentsDesiredY);
__global__ void scaleHeatmapCuda(int *heatmap_cuda, int *scaled_heatmap_cuda);
__global__ void blurHeatmapCuda(int *scaled_heatmap_cuda, int *blurred_heatmap_cuda);

