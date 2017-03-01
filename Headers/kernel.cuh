#pragma once

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
#include <chrono>



// Template structure to pass to kernel
template < typename T >
struct KernelArray
{
T* _array;
int _size;

// constructor allows for implicit conversion
KernelArray(thrust::device_vector<T>& dVec) {
    _array = thrust::raw_pointer_cast( &dVec[0] );
    _size  = ( int ) dVec.size();
  }
};


// Device Functions
__device__ double NextRate(double randomValue, const double alpha, const double beta,
                                      const double sigma, const double r0, const double cirFlag,
                                      const double sqrtDeltaT);

__host__ __device__ double getYield(const double tau, const double alpha, const double beta,
                              const double sigma, const double rNext, const double cirFlag);


// Global Functions
__global__ void initializeCurand(curandState * __restrict state, const unsigned long int seed);

__global__ void initializeNextRateRands(curandState * state, KernelArray<double> nextRateRands);

__global__ void initializePopulation(curandState * state,
                                      KernelArray<double> alphaRands, KernelArray<double> betaRands,
                                      KernelArray<double> sigmaRands, const int NP,
                                      const KernelArray<double> lowerBound, const KernelArray<double> upperBound);

__global__ void creatMutationIndexes(curandState * state,
                                    const int NP, KernelArray<int> mutIndx, KernelArray<double> randVal );

__global__ void mutateAndCrossOver(const int NP, const double CR, const double F,
                                  KernelArray<int> mutIndx, KernelArray<double> randVal,
                                  KernelArray<double> newAlpha, KernelArray<double> newBeta,
                                  KernelArray<double> newSigma,
                                  KernelArray<double> alpha, KernelArray<double> beta, KernelArray<double> sigma,
                                  const KernelArray<double> lowerBound, const KernelArray<double> upperBound);

__global__ void evaluateVasicek(KernelArray<double> crrntMonthMdlData, KernelArray<double> crrntMonthMrktData,
                                KernelArray<double> alpha, KernelArray<double> beta,
                                KernelArray<double> sigma, KernelArray<double> nextRateRands,
                                const int NP, double r0, KernelArray<double> dr, KernelArray<double> dr64,
                                KernelArray<double> rNext, KernelArray<double> tau, KernelArray<double> error,
                                const double cirFlag, const double sqrtDeltaT);

__global__ void selectMutatedOrOriginal(KernelArray<double> oldAlpha, KernelArray<double> oldBeta,
                                        KernelArray<double> oldSigma, KernelArray<double> newAlpha,
                                        KernelArray<double> newBeta,  KernelArray<double> newSigma,
                                        KernelArray<double> oldError,  KernelArray<double> newError,
                                        const int NP);
