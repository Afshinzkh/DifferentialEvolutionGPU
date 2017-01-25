// Helper functions Are declared and defined in this hedear
// functions are:
//          readData: Reads the Data from file
//          writeData: write the data to results.csv

#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <array>
#include <cstring>


  // e.g. M = 12 , N = 9
  template <size_t M, size_t N>
  void readData(std::string dataFileName,
      std::array< std::array <double, N>, M> &dataArray)
  {
    dataFileName = "../Data/" + dataFileName;
    std::cout << "file: " << dataFileName << std::endl;

    std::ifstream dataFile(dataFileName);
    int row = 0;
    int col = 0;
    if(dataFile.is_open())
    {
      std::string aLine;
      while(getline(dataFile, aLine))
      {
        std::istringstream ss(aLine);
        std::string num;
        while(ss >> num)
        {
          dataArray[row][col] = std::stod(num.c_str());
          col++;
        }
        row++;
        col = 0;
      }
    }
    dataFile.close();
  }

  template <size_t M, size_t N>
  void writeData(std::array< std::array <double, N>, M> &mdlArray,
          std::array< std::array <double, N>, M> &mrktArray,
          std::array< double, M> &alphaArray, std::array< double, M> &betaArray,
          std::array< double, M> &sigmaArray, std::array< double, M> &errorArray,
          std::array< double, M> &iterArray, std::array< double, M> &timeArray,
          std::string methodName)
  {

    std::string dataFileName = "../Data/" + methodName + ".csv";
    std::ofstream dataFile (dataFileName);

    dataFile << "Date;Alpha;Beta;Sigma;Error;Time;Iterations;\
    ModR0.25;ModR1;ModR3;ModR5;ModR7;ModR10;ModR15;ModR20;ModR30;\
    MarR0.25;MarR1;MarR3;MarR5;MarR7;MarR10;MarR15;MarR20;MarR30\n";

    for (size_t i = 0; i < M; i++) {

        dataFile << i+1 << ";";
        dataFile << alphaArray[i] << ";";
        dataFile << betaArray[i] << ";";
        dataFile << sigmaArray[i] << ";";
        dataFile << errorArray[i] << ";";
        dataFile << timeArray[i] << ";";
        dataFile << iterArray[i] << ";";
        for (size_t j = 0; j < N; j++)
          dataFile << mdlArray[i][j] << ";";
        for (size_t j = 0; j < N; j++)
            dataFile << mrktArray[11-i][j] << ";";
        dataFile << "\n ";
        // dataFile << "\n ";

        // dataFile << " Data for time-serie: " << i+1 << std::endl;
        // dataFile << "Model\n ";
        // dataFile << "0.25;1;3;5;7;10;15;20;30;\n ";
        // for (size_t j = 0; j < N; j++) {
        //   dataFile << mdlArray[i][j] << ";";}
        //
        // dataFile << "\n ";
        //
        //
        //
        // dataFile << "Market\n ";
        // dataFile << "0.25;1;3;5;7;10;15;20;30;\n ";
        // for (size_t j = 0; j < N; j++) {
        //   dataFile << mrktArray[11-i][j] << ";";}
        //
        // dataFile << "\n ";
        //
        // dataFile << "Final Parameters:\n";
        // dataFile << "Alpha;" << alphaArray[i] << ";;";
        // dataFile << "Beta;" << betaArray[i] << ";;";
        // dataFile << "Sigma;" << sigmaArray[i] << ";;";
        //
        // dataFile << "\n ";
        //
        // dataFile << "Iterations;" << iterArray[i] << ";;";
        // dataFile << "Error;" << errorArray[i] << ";;";
        // dataFile << "Time;" << timeArray[i] << ";;";
        //
        // dataFile << "\n ";
        // dataFile << "\n ";

        }
    dataFile.close();
  }

  template <size_t M, size_t N>
  void writeHullWhiteData(std::array< std::array <double, N>, M> &mdlArray,
          std::array< std::array <double, N>, M> &mrktArray,
          std::array< double, M> &alpha1Array,
          std::array< double, M> &sigma1Array,
          std::array< double, M> &alpha2Array,
          std::array< double, M> &sigma2Array,
          std::array< double, M> &rhoArray,
           std::array< double, M> &errorArray,
          std::array< double, M> &iterArray, std::array< double, M> &timeArray)
  {

    std::string dataFileName = "../Data/results.csv";
    std::ofstream dataFile (dataFileName);


    for (size_t i = 0; i < M; i++) {
        dataFile << "Date;Alpha1;Sigma1;Alpha2;Sigma2;Rho;\
        Error;Time;Iterations;\
  ModR0.25;ModR1;ModR3;ModR5;ModR7;ModR10;ModR15;ModR20;ModR30;\
  MarR0.25;MarR1;MarR3;MarR5;MarR7;MarR10;MarR15;MarR20;MarR30\n";

        dataFile << i+1 << ";";
        dataFile << alpha1Array[i] << ";";
        dataFile << sigma1Array[i] << ";";
        dataFile << alpha2Array[i] << ";";
        dataFile << sigma2Array[i] << ";";
        dataFile << rhoArray[i] << ";";
        dataFile << errorArray[i] << ";";
        dataFile << timeArray[i] << ";";
        dataFile << iterArray[i] << ";";
        for (size_t j = 0; j < N; j++)
          dataFile << mdlArray[i][j] << ";";
        for (size_t j = 0; j < N; j++)
            dataFile << mrktArray[11-i][j] << ";";
        dataFile << "\n ";
        dataFile << "\n ";

        }
    dataFile.close();
  }
