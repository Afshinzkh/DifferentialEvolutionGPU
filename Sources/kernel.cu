#include "../Headers/kernel.cuh"

__device__ double NextRate(double randomValue, const double alpha, const double beta,
                                      const double sigma, const double r0)
{

  double deltaT = 1.0 / 12.0;
  return alpha * (beta - r0) * deltaT + sigma * std::sqrt(deltaT) * randomValue;
}

__host__ __device__ double getYield(const double tau, const double alpha, const double beta,
                              const double sigma, const double rNext)
{
  double yield;
  double A,B,bondPrice;

  B = (1.0-std::exp(-alpha*tau))/alpha;
  A = std::exp(((B - tau)*(alpha * alpha * beta - 0.5*sigma * sigma)\
  /(alpha * alpha)) - (sigma*sigma*B*B/(4*alpha)));

  bondPrice = A*std::exp(-rNext*B);
  if (bondPrice == 0)	bondPrice = 0.000001;

  yield = ((-1.0/tau)*std::log(bondPrice));

  if(yield > 10000)	yield = 10;
  if(yield > 10000) yield = -10;

  // printf("A is %f\n", yield);
  return yield;
} //getYield



__global__ void initializeCurand(curandState * __restrict state, const unsigned long int seed)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	curand_init(seed, tid, 0, &state[tid]);
}


 __global__ void initializeNextRateRands(curandState * state, KernelArray<double> nextRateRands)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= 10000)
    return;

  nextRateRands._array[tid] = curand_normal(&state[tid*3]);
}


__global__ void initializePopulation(curandState * state,
                                      KernelArray<double> alphaRands, KernelArray<double> betaRands,
                                      KernelArray<double> sigmaRands, const int NP,
                                      const KernelArray<double> lowerBound, const KernelArray<double> upperBound)

{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= NP)
    return;

  alphaRands._array[tid] = (upperBound._array[0] - lowerBound._array[0]) * curand_uniform(&state[tid*3]) + lowerBound._array[0];
  betaRands._array[tid] = (upperBound._array[1] - lowerBound._array[1]) * curand_uniform(&state[tid*3+1]) + lowerBound._array[1];
  sigmaRands._array[tid] = (upperBound._array[2] - lowerBound._array[2]) * curand_uniform(&state[tid*3+2]) + lowerBound._array[2];
}

__global__ void creatMutationIndexes(curandState * state,
                                    const int NP, KernelArray<int> mutIndx, KernelArray<double> randVal )
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= NP)
      return;

    int a, b, c;

    do a = NP * (curand_uniform(&state[tid*3]));  while(a==tid);
    do b = NP * (curand_uniform(&state[tid*3]));  while(b==tid||b==a);
    do c = NP * (curand_uniform(&state[tid*3]));  while(c==tid||c==a||c==b);
    mutIndx._array[tid*3] = a;
    mutIndx._array[tid*3+1] = b;
    mutIndx._array[tid*3+2] = c;

    randVal._array[tid]=curand_uniform(&state[tid*3]);
}

__global__ void mutateAndCrossOver(const int NP, const double CR, const double F,
                                  KernelArray<int> mutIndx, KernelArray<double> randVal,
                                  KernelArray<double> newAlpha, KernelArray<double> newBeta,
                                  KernelArray<double> newSigma,
                                  KernelArray<double> alpha, KernelArray<double> beta, KernelArray<double> sigma,
                                  const KernelArray<double> lowerBound, const KernelArray<double> upperBound)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= NP)
      return;

      int a = mutIndx._array[tid*3];
      int b = mutIndx._array[tid*3+1];
      int c = mutIndx._array[tid*3+2];

      if(randVal._array[tid]<CR)
      {
          newAlpha._array[tid] = alpha._array[a] + F * (alpha._array[b] - alpha._array[c]);
          if(newAlpha._array[tid] > upperBound._array[0]) newAlpha._array[tid] = upperBound._array[0];
          else if (newAlpha._array[tid] < lowerBound._array[0]) newAlpha._array[tid] = lowerBound._array[0];

          newBeta._array[tid] = beta._array[a] + F * (beta._array[b] - beta._array[c]);
          if(newBeta._array[tid] > upperBound._array[1]) newBeta._array[tid] = upperBound._array[1];
          else if (newBeta._array[tid] < lowerBound._array[1]) newBeta._array[tid] = lowerBound._array[1];

          newSigma._array[tid] = sigma._array[a] + F * (sigma._array[b] - sigma._array[c]);
          if(newSigma._array[tid] > upperBound._array[2]) newSigma._array[tid] = upperBound._array[2];
          else if (newSigma._array[tid] < lowerBound._array[2]) newSigma._array[tid] = lowerBound._array[2];
      }
      else
      {
        newAlpha._array[tid] = alpha._array[tid];
        newBeta._array[tid] = beta._array[tid];
        newSigma._array[tid] = sigma._array[tid];
      }

}

__global__ void evaluateVasicek(KernelArray<double> crrntMonthMdlData, KernelArray<double> crrntMonthMrktData,
                                KernelArray<double> alpha, KernelArray<double> beta,
                                KernelArray<double> sigma, KernelArray<double> nextRateRands,
                                const int NP, double r0, KernelArray<double> dr, KernelArray<double> dr64,
                                KernelArray<double> rNext, KernelArray<double> tau, KernelArray<double> error)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= 640000)
    return;

  int tid2 = tid / 10000; // The index for Parameters
  int tid3 = tid % 10000; // The index for randomArray

  dr._array[tid] += NextRate(nextRateRands._array[tid3], alpha._array[tid2], beta._array[tid2], sigma._array[tid2], r0);
  __syncthreads();

  if (tid3 == 0) {

    for (int i = 0; i < 10000; ++i){ //TODO: use reduction instead
      dr64._array[tid2] += dr._array[tid2 * 10000 + i];
    }
  // printf("dr64 for tid %d is %f\n", tid2, r0 + dr64._array[tid2] / 10000 );
    rNext._array[tid2] = r0 + dr64._array[tid2] / 10000;

    // Now get yield
    for (int i = 0; i < 9; ++i)
        crrntMonthMdlData._array[i] = getYield(tau._array[i], alpha._array[tid2], beta._array[tid2], sigma._array[tid2], rNext._array[tid2]);

    // Now get Error
    for (int i = 0; i < 9; ++i)
		{
			error._array[tid2] += (crrntMonthMdlData._array[i] - crrntMonthMrktData._array[i])
									 	* (crrntMonthMdlData._array[i] - crrntMonthMrktData._array[i]);
		}
		error._array[tid2] = error._array[tid2]/9;
    // printf("Error here: %f\n", error._array[tid2] );
  }
  __syncthreads();



}



__global__ void selectMutatedOrOriginal(KernelArray<double> oldAlpha, KernelArray<double> oldBeta,
                                        KernelArray<double> oldSigma, KernelArray<double> newAlpha,
                                        KernelArray<double> newBeta,  KernelArray<double> newSigma,
                                        KernelArray<double> oldError,  KernelArray<double> newError,
                                        const int NP)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= NP)
    return;

  if(newError._array[tid] < oldError._array[tid])
  {
    oldAlpha._array[tid] = newAlpha._array[tid];
    oldBeta._array[tid] = newBeta._array[tid];
    oldSigma._array[tid] = newSigma._array[tid];
  }

}
