// #include "kernel.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <curand.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <vector>


__global__ void initializeCurand(curandState * state, const unsigned long int seed, const int mpCount)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.x * blockIdx.x + threadIdx.y;

	curand_init(seed, i*mpCount+j, 0, &state[i*mpCount + j]);
}

__global__ void initializePopulation(curandState * state, double* Population, const int NP, const int mpCount)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	// if (i < NP && j < mpCount)
	// {
	// 		Population[i * mpCount + j] = curand_uniform(&state[i * mpCount + j]);
	// }
	// for Test
	 if (i < NP && j < mpCount)
	{
			Population[i * mpCount + j] = (i*mpCount+j);
	}


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

void runDE()
{
  //set DE Variables
  const int NP = 64;
  const double F = 0.8;
  const double CR = 0.6;

	// Define General Variables
	dim3 threadsPerBlock = (64,3);
	// dim3 numBlocks = (1,1);
  // const int threads = 64; //TODO: think of a good number for blocks and threads
  // const int blocks =  16;

	// Define Generic Variables that later would be changeable
	int mpCount = 3; // This is 3 because CIR and vasicek both have 3 parameters
	std::vector<double> upperBound;
	std::vector<double> lowerBound;
	// for(int i = 0; i<mpCount; ++i)
	// {
	// 	lowerBound[i] = 0.00001;
	// 	upperBound[i] = 0.25;
	// }
	std::cout << "I'm Here" << '\n';
	// Define Host Variables
	thrust::host_vector < double > P(NP * mpCount);

	// Define Device Variables and pointers
	thrust::device_vector < double> dP = P;
	double *dPPointer = thrust::raw_pointer_cast(dP.data());

	// Initialize Curand and genererate the random populations
	curandState *dState;
	cudaMalloc(&dState, 64 * 3);
	initializeCurand <<< 1,threadsPerBlock >>> (dState , time(NULL), mpCount);
	initializePopulation <<< 1,threadsPerBlock >>> (dState, dPPointer, NP, mpCount);

	P = dP;
	for(int i = 0; i < P.size(); ++i)
		std::cout << "P  in locataion: " << i <<  " is " << P[i] << std::endl;

}

int main()
{
	std::vector<double> mrktData(10,1.0);
	std::cout << "Hi" << '\n';
	runDE();
	return 0;
}
