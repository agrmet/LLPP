#include "cuda_runtime.h"

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE *CELLSIZE

// For heatmapCuda: 
// heatmap_cuda is a 1D representation of the 2D heatmap.
int *heatmap_cuda;
int *scaled_heatmap_cuda;
int *blurred_heatmap_cuda;
cudaStream_t streamCuda;

// agent desired positions 
int* agentsDesiredX;
int* agentsDesiredY;
// had to add these also for some reason (they are just temporary arrays to move from cpu to gpu)
int* desXsCPU;
int* desYsCPU;

void setupHeatmapCuda();
void updateHeatmapCuda();

__global__ void fadeHeatmapCuda(int *heatmap_cuda);
__global__ void intensifyHeatmapCuda(int *heatmap_cuda, int *agentsDesiredX, int *agentsDesiredY);
__global__ void scaleHeatmapCuda();
__global__ void blurHeatmapCuda();

