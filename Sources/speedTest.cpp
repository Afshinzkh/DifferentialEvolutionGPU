#include "../Headers/DE.cuh"
#include "../Headers/Helper.h"

int main(int argc, char* argv[])
{
  // Cheking the Arguments
  if( argc != 3){
    std::cout << "Error: Wrong number of Arguments" << std::endl;
    return -1;
  }

  std::string method = argv[2];
  std::cout << "Method to use: "<< method << std::endl;
  // int theDates = argv[3];

  const int maturityCount = 9;
  const int seriesCount = 12;
  double tau[] = {0.25, 1, 3, 5, 7, 10, 15, 20, 30};


  std::array<std::array<double,9>, seriesCount> myData;
  readData(argv[1], myData);
  std::array<double,9> crrntMonthMrktDataArray;
  std::vector<double> crrntMonthMrktData;
  std::cout << "Data is Read." << '\n';



  std::array<double , seriesCount> alphaArray;
  std::array<double , seriesCount> betaArray;
  std::array<double , seriesCount> sigmaArray;
  std::array<double , seriesCount> errorArray;
  std::array<double , seriesCount> iterArray;
  std::array<double , seriesCount> timeArray;
  std::array< std::array< double, maturityCount>, seriesCount> mdlData;



  double deltaTTerm = std::sqrt(1.0/seriesCount);
  DE d(method, deltaTTerm);

  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i<seriesCount; i++){
    crrntMonthMrktDataArray = myData[seriesCount-1-i];
    crrntMonthMrktData.insert(crrntMonthMrktData.begin(), &crrntMonthMrktDataArray[0], &crrntMonthMrktDataArray[9]);
    d.setMrktArray(crrntMonthMrktData);
    d.runDE();
    alphaArray[i] = d.getAlpha();
    betaArray[i] = d.getBeta();
    sigmaArray[i] = d.getSigma();
    errorArray[i] = d.getError();
    mdlData[i] = d.getMdlArray();
    iterArray[i] = d.getIter();
    timeArray[i] = d.getTime();

  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> durationCount = end - start;
  double totalTime = durationCount.count();

/****************************************************************************/
/*************************** STEP 4 : Print Out *****************************/
/****************************************************************************/

  for(int i = 0; i < seriesCount; i++)
  {
    std::cout << "\nfinal alpha:" <<  alphaArray[i] <<std::endl;
    std::cout << "final beta:" << betaArray[i] <<std::endl;
    std::cout << "final sigma:" << sigmaArray[i] <<std::endl;
    std::cout << "Average Error for month : " << i;
    std::cout << "\t is : " << errorArray[i] << std::endl;
    std::cout << "Elapsed Time: " << timeArray[i] << std::endl;
    std::cout << "Number of Iterations: " << iterArray[i] << std::endl;
    for (size_t j = 0; j < 9; j++) {
      std::cout << "y for maturity: "  << tau[j] << "\t is: \t" << mdlData[i][j] << std::endl;
    }
  }

  method = "SpeedTest" + method + "GPU" ;
  writeData(mdlData, myData, alphaArray, betaArray, sigmaArray,
          errorArray, iterArray, timeArray,method);

  std::cout << "Data has been written to file" << std::endl;
  std::cout << "Total Calculation Time is: " << totalTime << std::endl;

  return 0;
}
