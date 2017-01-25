#include "../Headers/DE.cuh"


double DE::runDE(double r0)
{
  //set DE Variables
  const int NP = 64;
  const double F = 0.8;
  const double CR = 0.6;

  r0 = r0/3;;

  const int tau = 9;
  const int scenarioCount = 10000;
  // Define General Variables
  dim3 threadsPerBlock = 1024;
  dim3 numBlocks = 1024;

  // TODO: Define Generic Variables that later would be changeable
  int mpCount = 3; // This is 3 because CIR and vasicek both have 3 parameter
  double up[3] = {0.25, 0.05, 0.005};
  double lo[3] = {0.00001, 0.00001, 0.00001};
  thrust::device_vector <double> upperBound(up, up + 3);
  thrust::device_vector <double> lowerBound(lo, lo + 3);
  double maturityArray[] = {0.25, 1, 3, 5, 7, 10, 15, 20, 30};
  // Define Device Variables
  thrust::device_vector < double> alphaFinal(NP);
  thrust::device_vector < double> betaFinal(NP);
  thrust::device_vector < double> sigmaFinal(NP);
  thrust::device_vector < double> alphaNew(NP);
  thrust::device_vector < double> betaNew(NP);
  thrust::device_vector < double> sigmaNew(NP);

  thrust::device_vector < int > mutIndx(NP * 3);
  thrust::device_vector < double> mutRandVals(NP);

  thrust::device_vector < double> nextRateRands(scenarioCount);
  thrust::device_vector < double> deltaR(NP * scenarioCount);
  thrust::device_vector < double> deltaR64(NP);
  thrust::device_vector < double> rNext(NP);
  thrust::device_vector < double> maturity(maturityArray, maturityArray + 9);
  thrust::device_vector < double> errorFinal(NP);
  thrust::device_vector < double> errorNew(NP);


  thrust::device_vector < double> crrntMonthMdlData(tau);
  thrust::device_vector < double> crrntMonthMrktData = crrntMonthMrktDataVec;

  double errorAverage = 1.0;
  double tol = 0.0000008;
  gens = 1;

  auto start = std::chrono::steady_clock::now();

  // Initialize Curand and genererate the random populations
  curandState *dState;
  cudaMalloc(&dState, NP * mpCount * sizeof(curandState));
  initializeCurand <<< 512,512 >>> (dState , time(NULL));
  cudaThreadSynchronize();
  initializeNextRateRands <<< 512,512 >>> (dState, nextRateRands);
  initializePopulation <<< 16,16 >>> (dState, alphaFinal, betaFinal, sigmaFinal,
                                                                NP, lowerBound, upperBound);
  while(errorAverage > tol && gens < 50)
  {


    // std::cout << "alpha , beta, sigma:" << alphaFinal[0] << betaFinal[0] << sigmaFinal[0] << std::endl;
    creatMutationIndexes <<< 16,16 >>> (dState, NP, mutIndx, mutRandVals );
    cudaThreadSynchronize();
    // Reset The Fragile Vectors
    thrust::fill( deltaR.begin(), deltaR.end(), 0.0);
    thrust::fill( deltaR64.begin(), deltaR64.end(), 0.0);
    thrust::fill( errorFinal.begin(), errorFinal.end(), 0.0);


    evaluateVasicek <<< 1024,1024 >>> (crrntMonthMdlData, crrntMonthMrktData,
                                    alphaFinal, betaFinal, sigmaFinal, nextRateRands, NP, r0, deltaR, deltaR64,
                                    rNext, maturity, errorFinal);
    // check Tolerance
    errorAverage = thrust::reduce(errorFinal.begin(), errorFinal.end()) / errorFinal.size();
    std::cout << "average error for Generation " << gens << " is: "<< errorAverage << std::endl;
    gens++;

    cudaThreadSynchronize();
    mutateAndCrossOver <<< 16,16 >>> (NP, CR, F, mutIndx, mutRandVals, alphaNew, betaNew,
                                      sigmaNew, alphaFinal, betaFinal, sigmaFinal, lowerBound, upperBound);
    cudaThreadSynchronize();
    // Reset The Fragile Vectors
    thrust::fill( deltaR.begin(), deltaR.end(), 0.0);
    thrust::fill( deltaR64.begin(), deltaR64.end(), 0.0);
    thrust::fill( errorNew.begin(), errorNew.end(), 0.0);

    evaluateVasicek <<< 1024,1024 >>> (crrntMonthMdlData, crrntMonthMrktData,
                                    alphaNew, betaNew, sigmaNew, nextRateRands, NP, r0, deltaR, deltaR64,
                                    rNext, maturity, errorNew);
    cudaThreadSynchronize();
    selectMutatedOrOriginal <<< 16,16 >>> (alphaFinal, betaFinal, sigmaFinal, alphaNew, betaNew,
                                                               sigmaNew, errorFinal, errorNew, NP);
    cudaThreadSynchronize();
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> durationCount = end - start;
  calTime = durationCount.count();


  thrust::device_vector<double>::iterator iter =  thrust::min_element(errorFinal.begin(), errorFinal.end());
  unsigned int minErrorPosition = iter - errorFinal.begin();

  avgError = errorFinal[minErrorPosition] ;
  alpha = alphaFinal[minErrorPosition];
  beta = betaFinal[minErrorPosition];
  sigma = sigmaFinal[minErrorPosition];
  std::cout << "Final Error: " << avgError << std::endl;
  std::cout << "Final Alpha: " << alpha << std::endl;
  std::cout << "Final Beta: " << beta << std::endl;
  std::cout << "Final Sigma: " << sigma << std::endl;
  std::cout << "Calculation Time: " << calTime << std::endl;
  std::cout << "NewR: " << rNext[minErrorPosition] << std::endl;

  for (size_t i = 0; i < 9; i++)
    crrntMonthMdlDataArray[i] =  getYield(maturityArray[i], alpha, beta, sigma, rNext[minErrorPosition]);

  return rNext[minErrorPosition];

}

/****************************************************************************/
/******************** Setters and Getters are here **************************/
/****************************************************************************/

DE::DE(std::string m)
{
  methodName = m;
}

const double& DE::getAlpha() const { return alpha; }
const double& DE::getBeta() const { return beta; }
const double& DE::getSigma() const { return sigma; }
const double& DE::getError() const { return avgError; }
const int& DE::getIter() const { return gens; }
const double& DE::getTime() const { return calTime; }

const std::array<double, 9>& DE::getMdlArray() const
{
  return crrntMonthMdlDataArray;
}

void DE::setMrktArray(std::vector<double> const& mrktData)
{
  crrntMonthMrktDataVec = mrktData;
}