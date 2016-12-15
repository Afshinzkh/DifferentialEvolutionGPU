#include "kernel.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>

__global__ void curand_setup_kernel(curandState * __restrict state, const unsigned long int seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(seed, tid, 0, &state[tid]);
}

__global__ void initializePopulation()
{
  int int tid = blockIdx.x * blockDim.x + threadIdx.x;

}

__global__ void creatMutationIndexes()
{

}

__global__ void mutateAndCrossOver()
{

}

__global__ void evaluateVasicek()
{

}

__global__ void selectMutatedOrOriginal()
{

}

void runDE(std::vector<double> const& mrktData)
{
  //set DE Variables
  const int NP = 64;
  const double F = 0.8;
  const double CR = 0.6;


  const int threads = 64; //TODO: think of a good number for blocks and threads
  const int blocks =  16;
}
